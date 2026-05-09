import os
import json
import numpy as np
from typing import List, Dict, Optional
from config import config


class VectorStore:
    """Vector store wrapper supporting Chroma and FAISS backends."""

    def __init__(self, store_type: Optional[str] = None,
                 collection_name: Optional[str] = None):
        self.store_type = store_type or config.vector_store_type
        self.collection_name = collection_name or config.collection_name
        self._store = None
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        config.ensure_dirs()

    def _init_chroma(self):
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=config.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Delete if exists to allow fresh indexing
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._store = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self):
        self._faiss_index = None
        self._faiss_data: Dict = {"texts": [], "metadatas": [], "embeddings": None}

    def add(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add chunks with their embeddings to the store."""
        texts = [c["content"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"],
                       "strategy": c.get("strategy", ""), "id": c["id"]}
                     for c in chunks]
        ids = [c["id"] for c in chunks]

        if self.store_type == "chroma":
            if self._store is None:
                self._init_chroma()
            self._store.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        elif self.store_type == "faiss":
            if self._faiss_index is None:
                self._init_faiss()
            import faiss
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
            self._faiss_index.add(embeddings.astype(np.float32))
            self._faiss_data["texts"].extend(texts)
            self._faiss_data["metadatas"].extend(metadatas)
            self._faiss_data["ids"] = ids
            self._faiss_data["dim"] = dim
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")

        self._texts = texts
        self._metadatas = metadatas

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Query the store and return results with scores."""
        if self.store_type == "chroma":
            return self._query_chroma(query_embedding, top_k)
        elif self.store_type == "faiss":
            return self._query_faiss(query_embedding, top_k)
        return []

    def _query_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        results = self._store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i] / 2,  # Cosine distance to similarity
            })
        return out

    def _query_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        import faiss
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(q, top_k)
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_data["texts"]):
                continue
            out.append({
                "id": self._faiss_data["ids"][idx],
                "content": self._faiss_data["texts"][idx],
                "metadata": self._faiss_data["metadatas"][idx],
                "score": float(score),
            })
        return out

    def count(self) -> int:
        """Return the number of stored vectors."""
        if self.store_type == "chroma" and self._store is not None:
            return self._store.count()
        elif self.store_type == "faiss" and self._faiss_index is not None:
            return self._faiss_index.ntotal
        return 0

    @property
    def is_initialized(self) -> bool:
        if self.store_type == "chroma":
            return self._store is not None
        return self._faiss_index is not None
