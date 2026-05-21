import pytest
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def dummy_embeddings():
    """4 random normalized vectors of dim 384 (MiniLM size)."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(4, 384).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def dummy_chunks():
    return [
        {"id": f"doc_chunk_{i:04d}", "source": f"lecture_{i}.txt",
         "chunk_index": i, "content": f"This is test chunk number {i}.",
         "strategy": "test"}
        for i in range(4)
    ]


class TestChromaStore:
    def test_add_and_query(self, dummy_chunks, dummy_embeddings,
                           chroma_tmp_path, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        from src.vector_store import VectorStore
        store = VectorStore(store_type="chroma")
        store.add(dummy_chunks, dummy_embeddings)

        assert store.count() == 4

        query_vec = dummy_embeddings[0]
        results = store.query(query_vec, top_k=2)
        assert len(results) == 2
        for r in results:
            assert "id" in r
            assert "content" in r
            assert "metadata" in r
            assert "score" in r
        # The first result should be the exact match (highest similarity)
        assert results[0]["id"] == "doc_chunk_0000"

    def test_count_reflects_stored_vectors(self, dummy_chunks, dummy_embeddings,
                                           chroma_tmp_path, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        from src.vector_store import VectorStore
        store = VectorStore(store_type="chroma")
        assert store.count() == 0
        store.add(dummy_chunks, dummy_embeddings)
        assert store.count() == 4

    def test_clear_on_rebuild(self, dummy_chunks, dummy_embeddings,
                              chroma_tmp_path, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        from src.vector_store import VectorStore
        store = VectorStore(store_type="chroma")

        # First build: 4 vectors
        store.add(dummy_chunks, dummy_embeddings)
        assert store.count() == 4

        # Rebuild with only 2 vectors — should clear old data
        store2 = VectorStore(store_type="chroma")
        store2.add(dummy_chunks[:2], dummy_embeddings[:2])
        assert store2.count() == 2

    def test_is_initialized(self, dummy_chunks, dummy_embeddings,
                            chroma_tmp_path, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        from src.vector_store import VectorStore
        store = VectorStore(store_type="chroma")
        assert not store.is_initialized
        store.add(dummy_chunks, dummy_embeddings)
        assert store.is_initialized


class TestFAISSStore:
    @pytest.fixture(autouse=True)
    def check_faiss(self):
        pytest.importorskip("faiss")

    def test_add_and_query(self, dummy_chunks, dummy_embeddings):
        from src.vector_store import VectorStore
        store = VectorStore(store_type="faiss")
        store.add(dummy_chunks, dummy_embeddings)

        assert store.count() == 4

        query_vec = dummy_embeddings[0]
        results = store.query(query_vec, top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc_chunk_0000"

    def test_count(self, dummy_chunks, dummy_embeddings):
        from src.vector_store import VectorStore
        store = VectorStore(store_type="faiss")
        store.add(dummy_chunks, dummy_embeddings)
        assert store.count() == 4

    def test_is_initialized(self, dummy_chunks, dummy_embeddings):
        from src.vector_store import VectorStore
        store = VectorStore(store_type="faiss")
        assert not store.is_initialized
        store.add(dummy_chunks, dummy_embeddings)
        assert store.is_initialized


class TestVectorStoreErrors:
    def test_unknown_store_type_raises(self, chroma_tmp_path, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "chroma_dir", chroma_tmp_path)

        from src.vector_store import VectorStore
        store = VectorStore(store_type="invalid_backend")
        with pytest.raises(ValueError, match="Unknown store type"):
            store.add([], np.empty((0, 128)))

    def test_query_without_data_returns_empty(self):
        pytest.importorskip("faiss", reason="FAISS not installed")
        from src.vector_store import VectorStore
        store = VectorStore(store_type="faiss")
        vec = np.random.randn(128).astype(np.float32)
        results = store.query(vec, top_k=5)
        assert results == []
