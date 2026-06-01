import time
import logging
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from config import config
from src.pipeline import RAGPipeline


class Evaluator:
    """Evaluate RAG system performance: retrieval and generation quality."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.results: Dict = {}

    def evaluate_retrieval(self, queries: Optional[List[str]] = None) -> Dict:
        """Evaluate retrieval quality through self-consistency, score distribution,
        and latency. Without ground-truth relevance labels, MRR/Recall/NDCG are
        not computable — BM25-vector overlap serves as a consistency proxy instead.
        """
        if queries is None:
            queries = config.eval_queries

        all_results = []
        latencies = []
        bm25_vector_overlap = []
        score_variance = []

        for q in queries:
            t0 = time.time()
            hybrid = self.pipeline.retriever.retrieve(q)
            bm25 = self.pipeline.retriever.retrieve_bm25_only(q)
            vector = self.pipeline.retriever.retrieve_vector_only(q)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            hybrid_ids = set(r["id"] for r in hybrid)
            bm25_ids = set(r["id"] for r in bm25)
            vector_ids = set(r["id"] for r in vector)

            bm25_vector_overlap.append(
                len(bm25_ids & vector_ids) / max(len(bm25_ids | vector_ids), 1)
            )

            # Per-result scores: prefer rrf_score in RRF mode, fall back to score
            rrf_scores = [r.get("rrf_score", r.get("score", 0)) for r in hybrid]
            if rrf_scores:
                score_variance.append(float(np.var(rrf_scores)))

            all_results.append({
                "query": q,
                "num_hybrid": len(hybrid),
                "num_bm25": len(bm25),
                "num_vector": len(vector),
                "bm25_vector_jaccard": bm25_vector_overlap[-1],
                "hybrid_top_score": rrf_scores[0] if rrf_scores else 0,
                "hybrid_avg_top5_score": float(np.mean(rrf_scores[:5])) if rrf_scores else 0,
                "latency_ms": round(latency),
            })

        self.results["retrieval"] = {
            "num_queries": len(queries),
            "avg_latency_ms": round(np.mean(latencies)),
            "avg_bm25_vector_overlap": round(np.mean(bm25_vector_overlap), 4),
            "avg_score_variance": round(np.mean(score_variance), 6) if score_variance else 0,
            "per_query": all_results,
        }
        return self.results["retrieval"]

    def evaluate_chunking_strategies(self, queries: Optional[List[str]] = None) -> Dict:
        """Compare retrieval under different chunking strategies.

        Saves and restores the original pipeline state so the caller's index
        is not destroyed by the comparison.
        """
        if queries is None:
            queries = config.eval_queries

        # Save original state
        _orig_chunker = self.pipeline.chunker
        _orig_store = self.pipeline.vector_store
        _orig_retriever = self.pipeline.retriever
        _orig_chunks = list(self.pipeline.chunks)
        _orig_strategy = self.pipeline.strategy

        comparison = {}
        docs = self.pipeline.loader.load_processed()
        if not docs:
            docs = self.pipeline.loader.run()

        try:
            for strategy in ["fixed_token", "recursive_char", "semantic"]:
                logging.getLogger("rag_lab").info(
                    "  Evaluating strategy: %s", strategy)
                self.pipeline.rebuild_with_strategy(strategy, docs)
                eval_result = self.evaluate_retrieval(queries)
                comparison[strategy] = {
                    "num_chunks": len(self.pipeline.chunks),
                    "avg_chunk_length": np.mean([c["length"] for c in self.pipeline.chunks])
                    if self.pipeline.chunks else 0,
                    "avg_latency_ms": eval_result["avg_latency_ms"],
                    "avg_bm25_vector_overlap": eval_result["avg_bm25_vector_overlap"],
                }
        finally:
            # Restore original state
            self.pipeline.chunker = _orig_chunker
            self.pipeline.vector_store = _orig_store
            self.pipeline.retriever = _orig_retriever
            self.pipeline.chunks = _orig_chunks
            self.pipeline.strategy = _orig_strategy

        self.results["chunking_comparison"] = comparison
        return comparison

    def evaluate_embedding_models(self) -> Dict:
        """Compare embedding model performance on retrieval."""
        docs = self.pipeline.loader.load_processed()
        if not docs:
            docs = self.pipeline.loader.run()

        chunks = self.pipeline.chunker.chunk_documents(docs)
        texts = [c["content"] for c in chunks[:20]]  # Sample for speed

        from src.embedder import compare_embedding_models
        query = config.eval_queries[0]  # Use config query, not hardcoded
        comparison = compare_embedding_models(texts, query)
        self.results["embedding_comparison"] = comparison
        return comparison

    def evaluate_generation_quality(self, queries: Optional[List[str]] = None) -> Dict:
        """Evaluate end-to-end generation quality using the configured LLM."""
        if queries is None:
            queries = config.eval_queries[:3]

        gen_results = []
        for q in queries:
            try:
                result = self.pipeline.query(q, use_llm=True)
            except Exception as e:
                result = {"answer": f"[Error: {e}]", "retrieved_docs": [],
                          "sources": [], "model": "error",
                          "retrieval_time_ms": 0}

            gen_results.append({
                "query": q,
                "answer_length": len(result.get("answer", "")),
                "num_sources": len(result.get("sources", [])),
                "sources": result.get("sources", []),
                "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            })

        self.results["generation"] = {
            "num_queries": len(queries),
            "avg_answer_length": np.mean([r["answer_length"] for r in gen_results]),
            "avg_sources_used": np.mean([r["num_sources"] for r in gen_results]),
            "per_query": gen_results,
        }
        return self.results["generation"]

    def run_full_evaluation(self) -> Dict:
        """Run all evaluations and return comprehensive results."""
        _log = logging.getLogger("rag_lab")
        _log.info("=" * 60)
        _log.info("RAG SYSTEM EVALUATION")
        _log.info("=" * 60)

        _log.info("[1/4] Evaluating retrieval quality...")
        self.evaluate_retrieval()
        _log.info("  Avg latency: %sms",
                   self.results['retrieval']['avg_latency_ms'])
        _log.info("  Avg BM25-Vector overlap: %s",
                   self.results['retrieval']['avg_bm25_vector_overlap'])

        _log.info("[2/4] Comparing chunking strategies...")
        self.evaluate_chunking_strategies()
        for s, r in self.results["chunking_comparison"].items():
            _log.info("  %s: %s chunks, latency=%sms",
                       s, r['num_chunks'], r['avg_latency_ms'])

        _log.info("[3/4] Comparing embedding models...")
        self.evaluate_embedding_models()

        _log.info("[4/4] Evaluating generation quality...")
        self.evaluate_generation_quality()

        return self.results

    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            self.run_full_evaluation()

        _log = logging.getLogger("rag_lab")
        _log.info("=" * 60)
        _log.info("EVALUATION REPORT")
        _log.info("=" * 60)

        if "retrieval" in self.results:
            r = self.results["retrieval"]
            _log.info("--- Retrieval Performance ---")
            _log.info("  Queries evaluated: %s", r['num_queries'])
            _log.info("  Avg latency: %sms", r['avg_latency_ms'])
            _log.info("  Avg BM25-Vector overlap: %s",
                       r['avg_bm25_vector_overlap'])

        if "chunking_comparison" in self.results:
            _log.info("--- Chunking Strategy Comparison ---")
            for name, data in self.results["chunking_comparison"].items():
                _log.info("  %-20s: chunks=%4d, avg_len=%.0f, latency=%sms",
                           name, data['num_chunks'],
                           data['avg_chunk_length'], data['avg_latency_ms'])

        if "embedding_comparison" in self.results:
            _log.info("--- Embedding Model Comparison ---")
            for model, data in self.results["embedding_comparison"].items():
                disc = data.get("discrimination", "N/A")
                _log.info("  %s: dim=%s, avg_sim=%.4f, discrimination=%s",
                           model, data['dim'], data['avg_similarity'], disc)

        if "generation" in self.results:
            g = self.results["generation"]
            _log.info("--- Generation Quality ---")
            _log.info("  Avg answer length: %.0f chars", g['avg_answer_length'])
            _log.info("  Avg sources per query: %.1f", g['avg_sources_used'])
