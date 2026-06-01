import logging
import numpy as np
from typing import List, Dict, Optional
from config import config


class Embedder:
    """Embedding model wrapper supporting multiple sentence-transformer models."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.default_embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logging.getLogger("rag_lab").info(
                "Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32,
              show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts and return numpy array (N x dim)."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, returning 1D numpy array."""
        return self.embed([query], show_progress=False)[0]


def compare_embedding_models(texts: List[str], query: str,
                             model_names: Optional[List[str]] = None):
    """Compare different embedding models on a set of texts."""
    if model_names is None:
        model_names = config.embedding_models

    results = {}
    available = []
    for mn in model_names:
        try:
            emb = Embedder(mn)
            _ = emb.embed(["test"], show_progress=False)
            available.append(mn)
        except Exception as e:
            logging.getLogger("rag_lab").warning(
                "Model %s not available: %s", mn, e)

    for mn in available:
        emb = Embedder(mn)
        doc_embs = emb.embed(texts, show_progress=False)
        query_emb = emb.embed_query(query)

        # Cosine similarity (already normalized)
        scores = np.dot(doc_embs, query_emb)
        top_indices = np.argsort(scores)[::-1][:5]

        mean_score = float(np.mean(scores))
        top_score = float(scores[top_indices[0]])

        results[mn] = {
            "dim": emb.dim,
            "top_scores": scores[top_indices].tolist(),
            "top_texts_preview": [texts[i][:100] for i in top_indices],
            "avg_similarity": mean_score,
            "top_score": top_score,
            "discrimination": round(top_score / (mean_score + 1e-8), 2),
        }

    return results
