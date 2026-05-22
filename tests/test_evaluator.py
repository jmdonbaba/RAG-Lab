import pytest
import numpy as np
from unittest.mock import MagicMock

from src.evaluator import Evaluator


@pytest.fixture
def mock_pipeline():
    """Pipeline with mocked components so we can test evaluator logic in isolation."""
    pipeline = MagicMock()
    pipeline.chunks = [
        {"id": f"doc1_chunk_{i:04d}", "source": "doc1.pdf", "chunk_index": i,
         "content": f"Content chunk {i}", "length": 100 + i * 10, "strategy": "test"}
        for i in range(10)
    ]
    pipeline.strategy = "recursive_char"

    # Mock retriever to return consistent results
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {"id": f"doc1_chunk_{i:04d}", "content": f"Content {i}",
         "metadata": {"source": "doc1.pdf", "chunk_index": i},
         "score": 1.0 - i * 0.1, "rrf_score": 0.05 - i * 0.005, "source": "fused"}
        for i in range(5)
    ]
    retriever.retrieve_bm25_only.return_value = [
        {"id": f"doc1_chunk_{i:04d}", "content": f"Content {i}",
         "metadata": {"source": "doc1.pdf", "chunk_index": i},
         "score": 10.0 - i, "source": "bm25"}
        for i in range(3)
    ]
    retriever.retrieve_vector_only.return_value = [
        {"id": f"doc1_chunk_{j:04d}", "content": f"Content {j}",
         "metadata": {"source": "doc1.pdf", "chunk_index": j},
         "score": 0.9 - j * 0.1, "source": "vector"}
        for j in [0, 2, 4]
    ]
    pipeline.retriever = retriever
    return pipeline


class TestEvaluateRetrieval:
    def test_result_structure_and_per_query_fields(self, mock_pipeline):
        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_retrieval(queries=["test query"])
        for key in ["num_queries", "avg_latency_ms", "avg_bm25_vector_overlap",
                     "avg_score_variance", "per_query"]:
            assert key in result
        assert result["num_queries"] == 1
        pq = result["per_query"][0]
        for key in ["bm25_vector_jaccard", "hybrid_top_score",
                     "hybrid_avg_top5_score", "latency_ms"]:
            assert key in pq

    def test_hybrid_top_score_uses_rrf_when_present(self, mock_pipeline):
        """When rrf_score is present, hybrid_top_score should come from rrf_score."""
        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_retrieval(queries=["test query"])
        top = result["per_query"][0]["hybrid_top_score"]
        # rrf_score of first result is 0.05, not the raw score 1.0
        assert top == pytest.approx(0.05)

    def test_handles_empty_retrieval(self, mock_pipeline):
        mock_pipeline.retriever.retrieve.return_value = []
        mock_pipeline.retriever.retrieve_bm25_only.return_value = []
        mock_pipeline.retriever.retrieve_vector_only.return_value = []
        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_retrieval(queries=["test query"])
        pq = result["per_query"][0]
        assert pq["bm25_vector_jaccard"] == 0.0
        assert pq["hybrid_top_score"] == 0.0

    def test_multiple_queries_aggregates_correctly(self, mock_pipeline):
        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_retrieval(queries=["q1", "q2", "q3"])
        assert result["num_queries"] == 3
        assert len(result["per_query"]) == 3


class TestEvaluateChunkingStrategies:
    def test_restores_pipeline_state(self, mock_pipeline):
        """After chunking comparison, pipeline should be restored to original state."""
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]

        orig_chunker = mock_pipeline.chunker
        orig_store = mock_pipeline.vector_store
        orig_retriever = mock_pipeline.retriever
        orig_strategy = mock_pipeline.strategy

        evaluator = Evaluator(mock_pipeline)
        evaluator.evaluate_chunking_strategies(queries=["test"])

        assert mock_pipeline.chunker is orig_chunker
        assert mock_pipeline.vector_store is orig_store
        assert mock_pipeline.retriever is orig_retriever
        assert mock_pipeline.strategy == orig_strategy

    def test_returns_all_strategies(self, mock_pipeline):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]

        evaluator = Evaluator(mock_pipeline)
        comparison = evaluator.evaluate_chunking_strategies(queries=["test"])

        for strat in ["fixed_token", "recursive_char", "semantic"]:
            assert strat in comparison
            assert "num_chunks" in comparison[strat]
            assert "avg_chunk_length" in comparison[strat]
            assert "avg_latency_ms" in comparison[strat]


class TestEvaluateEmbeddingModels:
    def test_uses_config_query_and_returns_comparison(self, mock_pipeline, monkeypatch):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content " * 10}
        ]
        mock_pipeline.chunker.chunk_documents.return_value = [
            {"id": f"c{i}", "source": "doc1.pdf", "content": f"content {i}",
             "length": 100, "strategy": "test"}
            for i in range(5)
        ]

        fake_comparison = {
            "all-MiniLM-L6-v2": {"dim": 384, "avg_similarity": 0.5,
                                  "top_score": 0.9, "discrimination": 1.8,
                                  "top_scores": [0.9, 0.8], "top_texts_preview": ["a", "b"]},
        }
        monkeypatch.setattr("src.embedder.compare_embedding_models",
                            lambda texts, query: fake_comparison)

        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_embedding_models()
        assert "all-MiniLM-L6-v2" in result
        assert result["all-MiniLM-L6-v2"]["dim"] == 384
        assert "embedding_comparison" in evaluator.results


class TestEvaluateGenerationQuality:
    def test_calls_pipeline_with_use_llm_true(self, mock_pipeline):
        mock_pipeline.query.return_value = {
            "answer": "Logistic regression is a classification algorithm.",
            "sources": ["logistic_regression.txt"],
            "model": "deepseek-chat",
            "retrieval_time_ms": 150,
            "retrieved_docs": [],
        }

        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_generation_quality(queries=["What is logistic regression?"])

        call_args = mock_pipeline.query.call_args
        assert call_args[1]["use_llm"] is True

        assert result["num_queries"] == 1
        assert result["avg_answer_length"] > 0
        assert result["avg_sources_used"] == 1
        pq = result["per_query"][0]
        assert pq["answer_length"] > 0
        assert "logistic_regression.txt" in pq["sources"]

    def test_handles_query_exception_gracefully(self, mock_pipeline):
        mock_pipeline.query.side_effect = RuntimeError("API unavailable")

        evaluator = Evaluator(mock_pipeline)
        result = evaluator.evaluate_generation_quality(queries=["test"])

        pq = result["per_query"][0]
        # answer_length = len("[Error: API unavailable]") — should be > 0
        assert pq["answer_length"] > 0
        assert pq["num_sources"] == 0
        assert result["avg_answer_length"] > 0


class TestRunFullEvaluation:
    def test_calls_all_four_evaluation_stages(self, mock_pipeline, monkeypatch):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]

        # Stub out the embedding comparison to avoid real model loading
        monkeypatch.setattr("src.embedder.compare_embedding_models",
                            lambda texts, query: {"mock-model": {"dim": 128,
                                  "avg_similarity": 0.5, "top_score": 0.9, "discrimination": 1.8,
                                  "top_scores": [], "top_texts_preview": []}})

        evaluator = Evaluator(mock_pipeline)
        results = evaluator.run_full_evaluation()

        assert "retrieval" in results
        assert "chunking_comparison" in results
        assert "embedding_comparison" in results
        assert "generation" in results

    def test_returns_same_dict_as_results_attribute(self, mock_pipeline, monkeypatch):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]
        monkeypatch.setattr("src.embedder.compare_embedding_models",
                            lambda texts, query: {"mock-model": {"dim": 128,
                                  "avg_similarity": 0.5, "top_score": 0.9, "discrimination": 1.8,
                                  "top_scores": [], "top_texts_preview": []}})

        evaluator = Evaluator(mock_pipeline)
        results = evaluator.run_full_evaluation()
        assert results is evaluator.results


class TestPrintReport:
    def test_prints_all_sections(self, mock_pipeline, monkeypatch, capsys):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]
        monkeypatch.setattr("src.embedder.compare_embedding_models",
                            lambda texts, query: {"mock-model": {"dim": 128,
                                  "avg_similarity": 0.5, "top_score": 0.9, "discrimination": 1.8,
                                  "top_scores": [], "top_texts_preview": []}})

        evaluator = Evaluator(mock_pipeline)
        evaluator.print_report()

        captured = capsys.readouterr().out
        assert "Retrieval Performance" in captured
        assert "Chunking Strategy Comparison" in captured
        assert "Embedding Model Comparison" in captured
        assert "Generation Quality" in captured

    def test_runs_evaluation_if_not_yet_run(self, mock_pipeline, monkeypatch, capsys):
        mock_pipeline.loader = MagicMock()
        mock_pipeline.loader.load_processed.return_value = [
            {"source": "doc1.pdf", "content": "Some content"}
        ]
        monkeypatch.setattr("src.embedder.compare_embedding_models",
                            lambda texts, query: {"mock-model": {"dim": 128,
                                  "avg_similarity": 0.5, "top_score": 0.9, "discrimination": 1.8,
                                  "top_scores": [], "top_texts_preview": []}})

        evaluator = Evaluator(mock_pipeline)
        assert evaluator.results == {}
        evaluator.print_report()
        assert len(evaluator.results) == 4
