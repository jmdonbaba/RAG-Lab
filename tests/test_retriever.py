import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.retriever import HybridRetriever, _has_chinese, _tokenize


# ------- Pure function tests ------------------------------------------------

class TestHasChinese:
    @pytest.mark.parametrize("text,expected", [
        ("Hello World", False),
        ("机器学习", True),
        ("Hello 世界", True),
        ("", False),
        ("12345 !@#$%", False),
        ("自然语言处理 NLP", True),
    ])
    def test_has_chinese(self, text, expected):
        assert _has_chinese(text) == expected


class TestTokenize:
    def test_english_splits_on_whitespace_and_lowercases(self):
        tokens = _tokenize("Hello World FROM Python")
        assert tokens == ["hello", "world", "from", "python"]

    def test_chinese_uses_jieba(self):
        tokens = _tokenize("机器学习很有趣")
        assert len(tokens) >= 2
        # jieba should segment this into meaningful words
        assert "机器" in tokens or "学习" in tokens or "有趣" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []


# ------- Helpers for fusion tests -------------------------------------------

def _make_result(doc_id, score, source="bm25"):
    return {
        "id": doc_id,
        "content": f"Content for {doc_id}",
        "metadata": {"source": f"{doc_id}.txt", "chunk_index": 0},
        "score": score,
        "source": source,
    }


def _make_mock_retriever(bm25_weight=0.2, vector_weight=0.8):
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(128)
    store = MagicMock()
    store.query.return_value = []
    return HybridRetriever(embedder, store, top_k=5,
                           bm25_weight=bm25_weight, vector_weight=vector_weight)


# ------- RRF fusion tests ---------------------------------------------------

class TestRRFFusion:
    def test_rrf_ranks_higher_when_doc_in_both_lists(self):
        ret = _make_mock_retriever(bm25_weight=0.5, vector_weight=0.5)
        bm25 = [_make_result("A", 10, "bm25"), _make_result("B", 5, "bm25")]
        vec = [_make_result("A", 0.9, "vector"), _make_result("C", 0.7, "vector")]
        fused = ret._rrf_fusion(bm25, vec)
        # "A" appears in both → should rank #1
        assert fused[0]["id"] == "A"

    def test_rrf_applies_correct_formula(self):
        ret = _make_mock_retriever(bm25_weight=0.2, vector_weight=0.8)
        bm25 = [_make_result("X", 10, "bm25")]
        vec = [_make_result("X", 0.9, "vector")]
        fused = ret._rrf_fusion(bm25, vec)
        k = 60
        expected = 0.2 / (k + 1) + 0.8 / (k + 1)
        assert fused[0]["rrf_score"] == pytest.approx(expected)

    def test_rrf_respects_top_k(self):
        ret = _make_mock_retriever()
        ret.top_k = 3
        bm25 = [_make_result(str(i), 10 - i, "bm25") for i in range(10)]
        vec = [_make_result(str(i), 1.0 - 0.1 * i, "vector") for i in range(5, 15)]
        fused = ret._rrf_fusion(bm25, vec)
        assert len(fused) == 3

    def test_rrf_empty_inputs(self):
        ret = _make_mock_retriever()
        fused = ret._rrf_fusion([], [])
        assert fused == []


# ------- Weighted fusion tests ----------------------------------------------

class TestWeightedFusion:
    def test_weighted_fusion_combines_scores(self):
        ret = _make_mock_retriever(bm25_weight=0.3, vector_weight=0.7)
        bm25 = [_make_result("D1", 10, "bm25")]
        vec = [_make_result("D1", 1.0, "vector")]
        fused = ret._weighted_fusion(bm25, vec)
        # Both normalized to 1.0, combined = 0.3*1.0 + 0.7*1.0 = 1.0
        assert fused[0]["score"] == pytest.approx(1.0)

    def test_weighted_fusion_doc_in_one_list_only(self):
        ret = _make_mock_retriever(bm25_weight=0.5, vector_weight=0.5)
        bm25 = [_make_result("only_bm25", 10, "bm25")]
        vec = [_make_result("only_vector", 1.0, "vector")]
        fused = ret._weighted_fusion(bm25, vec)
        ids = {r["id"] for r in fused}
        assert "only_bm25" in ids
        assert "only_vector" in ids
        # only_bm25 gets 0.5*1.0 + 0 = 0.5, only_vector gets 0 + 0.5*1.0 = 0.5
        for r in fused:
            assert r["score"] == pytest.approx(0.5)

    def test_weighted_fusion_respects_top_k(self):
        ret = _make_mock_retriever()
        ret.top_k = 4
        bm25 = [_make_result(f"B{i}", 10 - i, "bm25") for i in range(8)]
        vec = [_make_result(f"V{i}", 1.0 - 0.1 * i, "vector") for i in range(8)]
        fused = ret._weighted_fusion(bm25, vec)
        assert len(fused) == 4


# ------- HybridRetriever integration tests ----------------------------------

class TestHybridRetrieverIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, sample_docs, sample_chunks, embedder, chroma_tmp_path, monkeypatch):
        from config import config
        from src.vector_store import VectorStore

        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        self.store = VectorStore(store_type="chroma")
        texts = [c["content"] for c in sample_chunks]
        embs = embedder.embed(texts, show_progress=False)
        self.store.add(sample_chunks, embs)

        self.retriever = HybridRetriever(embedder, self.store, top_k=3)
        self.retriever.build_bm25(sample_chunks)

    def test_bm25_finds_exact_keywords(self):
        results = self.retriever.retrieve_bm25_only("logistic regression")
        assert len(results) >= 1
        assert any("logistic" in r["content"].lower() for r in results)

    def test_bm25_matches_on_keywords(self):
        """BM25 matches on token overlap, not semantic meaning — that's the vector's job."""
        results = self.retriever.retrieve_bm25_only("classification model optimization")
        # BM25 should return results via partial keyword matches (e.g. "model")
        assert len(results) > 0

    def test_vector_retrieval_finds_semantic_matches(self, embedder):
        """Vector search should return results even for queries with no keyword overlap."""
        results = self.retriever.retrieve_vector_only(
            "finding the best line to separate data points"
        )
        assert len(results) >= 1

    def test_hybrid_retrieval_returns_top_k(self):
        results = self.retriever.retrieve("gradient descent")
        assert len(results) == 3

    def test_all_retrieval_methods_return_results(self):
        bm25 = self.retriever.retrieve_bm25_only("optimization")
        vec = self.retriever.retrieve_vector_only("optimization")
        hybrid = self.retriever.retrieve("optimization")
        assert len(bm25) > 0
        assert len(vec) > 0
        assert len(hybrid) > 0


# ------- Query translation (unit) -------------------------------------------

class TestQueryTranslation:
    def test_translation_cache(self, monkeypatch):
        """Verify _translate_query caches results and doesn't re-call API."""
        from src.retriever import _translate_query, _translation_cache

        cache_before = dict(_translation_cache)

        # Inject a value directly into the cache
        test_query = "缓存测试查询"
        _translation_cache[test_query] = "cached test query"

        result = _translate_query(test_query)
        assert result == "cached test query"

        # Restore
        _translation_cache.clear()
        _translation_cache.update(cache_before)
