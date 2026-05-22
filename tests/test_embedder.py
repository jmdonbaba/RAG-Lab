import pytest
import numpy as np


@pytest.mark.usefixtures("embedder")
class TestEmbedder:
    def test_embed_query_returns_1d_array(self, embedder):
        vec = embedder.embed_query("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == embedder.dim

    def test_embed_returns_2d_array(self, embedder):
        texts = ["first sentence", "second sentence", "third sentence"]
        embs = embedder.embed(texts, show_progress=False)
        assert isinstance(embs, np.ndarray)
        assert embs.ndim == 2
        assert embs.shape == (3, embedder.dim)

    def test_embed_single_text_works(self, embedder):
        embs = embedder.embed("single text", show_progress=False)
        assert embs.shape == (1, embedder.dim)

    def test_embeddings_are_l2_normalized(self, embedder):
        texts = ["alpha", "beta", "gamma"]
        embs = embedder.embed(texts, show_progress=False)
        norms = np.linalg.norm(embs, axis=1)
        # normalize_embeddings=True should produce unit vectors
        assert np.allclose(norms, 1.0, atol=1e-5), f"norms: {norms}"

    def test_same_text_produces_same_embedding(self, embedder):
        text = "what is machine learning"
        v1 = embedder.embed_query(text)
        v2 = embedder.embed_query(text)
        assert np.allclose(v1, v2, atol=1e-6)

    def test_different_texts_produce_different_embeddings(self, embedder):
        v1 = embedder.embed_query("linear regression minimizes MSE")
        v2 = embedder.embed_query("convolutional neural networks for images")
        unrelated_sim = np.dot(v1, v2)

        # Related texts should have higher similarity than unrelated ones
        v3 = embedder.embed_query("linear models for prediction with least squares")
        related_sim = np.dot(v1, v3)
        assert related_sim > unrelated_sim
