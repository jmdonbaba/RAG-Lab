import numpy as np
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
import jieba

from config import config
from src.embedder import Embedder
from src.vector_store import VectorStore


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25, using jieba for Chinese and whitespace for English."""
    has_chinese = any('一' <= c <= '鿿' for c in text)
    if has_chinese:
        return list(jieba.cut(text))
    return text.lower().split()


class HybridRetriever:
    """
    混合检索器: BM25 关键词 + 向量语义 → RRF/加权融合

    为什么需要混合检索？
      - BM25 擅长精确关键词匹配（"logistic regression" 中的 "logistic"）
      - 向量检索擅长语义理解（"梯度下降" → "gradient descent" 同义）
      - 混合检索取两者之长，互补盲区

    融合策略:
      - RRF (Reciprocal Rank Fusion): 基于排名融合，不依赖分数绝对值
        RRF_score(d) = Σ w_i / (k + rank_i(d)), k=60
      - Weighted: 分数归一化后加权求和
        combined = w_bm25 * norm_bm25 + w_vector * norm_vector
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 top_k: int = None, bm25_weight: float = None, vector_weight: float = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k or config.top_k
        self.bm25_weight = bm25_weight if bm25_weight is not None else config.retrieval_weights["bm25"]
        self.vector_weight = vector_weight if vector_weight is not None else config.retrieval_weights["vector"]
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Dict] = []

    def build_bm25(self, chunks: List[Dict]):
        """Build BM25 index from chunks."""
        self._bm25_chunks = chunks
        tokenized = [_tokenize(c["content"]) for c in chunks]
        self._bm25_index = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> List[Dict]:
        """Perform hybrid retrieval and return top_k results with fused scores."""
        bm25_results = self._retrieve_bm25(query)
        vector_results = self._retrieve_vector(query)
        return self._fuse_results(bm25_results, vector_results)

    def _retrieve_bm25(self, query: str) -> List[Dict]:
        if self._bm25_index is None:
            return []

        tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:self.top_k * 2]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self._bm25_chunks[idx]["id"],
                    "content": self._bm25_chunks[idx]["content"],
                    "metadata": {
                        "source": self._bm25_chunks[idx]["source"],
                        "chunk_index": self._bm25_chunks[idx]["chunk_index"],
                    },
                    "score": float(scores[idx]),
                    "source": "bm25",
                })
        return results

    def _retrieve_vector(self, query: str) -> List[Dict]:
        q_emb = self.embedder.embed_query(query)
        results = self.vector_store.query(q_emb, top_k=self.top_k * 2)
        for r in results:
            r["source"] = "vector"
        return results

    def _fuse_results(self, bm25_results: List[Dict],
                      vector_results: List[Dict]) -> List[Dict]:
        """Fuse results using Reciprocal Rank Fusion (RRF) or weighted combination."""
        if config.hybrid_fusion == "rrf":
            return self._rrf_fusion(bm25_results, vector_results)
        else:
            return self._weighted_fusion(bm25_results, vector_results)

    def _rrf_fusion(self, bm25_results: List[Dict],
                    vector_results: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion."""
        scores: Dict[str, dict] = {}
        for rank, r in enumerate(bm25_results):
            rid = r["id"]
            if rid not in scores:
                scores[rid] = r.copy()
                scores[rid]["rrf_score"] = self.bm25_weight / (k + rank + 1)
            else:
                scores[rid]["rrf_score"] += self.bm25_weight / (k + rank + 1)

        for rank, r in enumerate(vector_results):
            rid = r["id"]
            if rid not in scores:
                scores[rid] = r.copy()
                scores[rid]["rrf_score"] = self.vector_weight / (k + rank + 1)
            else:
                scores[rid]["rrf_score"] += self.vector_weight / (k + rank + 1)

        merged = list(scores.values())
        merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return merged[:self.top_k]

    def _weighted_fusion(self, bm25_results: List[Dict],
                         vector_results: List[Dict]) -> List[Dict]:
        """Weighted linear combination of normalized scores."""
        # Normalize BM25 scores
        bm25_max = max((r["score"] for r in bm25_results), default=1.0)
        bm25_lookup = {r["id"]: r for r in bm25_results}

        # Normalize vector scores
        vec_max = max((r["score"] for r in vector_results), default=1.0)
        vec_lookup = {r["id"]: r for r in vector_results}

        all_ids = set(bm25_lookup.keys()) | set(vec_lookup.keys())
        fused = []
        for rid in all_ids:
            bm25_norm = bm25_lookup[rid]["score"] / bm25_max if rid in bm25_lookup else 0
            vec_norm = vec_lookup[rid]["score"] / vec_max if rid in vec_lookup else 0
            combined = self.bm25_weight * bm25_norm + self.vector_weight * vec_norm
            base = bm25_lookup.get(rid) or vec_lookup.get(rid)
            entry = base.copy()
            entry["score"] = combined
            entry["source"] = "fused"
            fused.append(entry)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:self.top_k]

    def retrieve_bm25_only(self, query: str) -> List[Dict]:
        """Retrieve using only BM25 for comparison."""
        return self._retrieve_bm25(query)[:self.top_k]

    def retrieve_vector_only(self, query: str) -> List[Dict]:
        """Retrieve using only vector similarity for comparison."""
        return self._retrieve_vector(query)[:self.top_k]
