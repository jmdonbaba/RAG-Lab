import json
import time
from typing import List, Dict, Optional
from dataclasses import asdict

from config import config
from src.document_loader import DocumentLoader
from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import HybridRetriever
from src.generator import Generator


class RAGPipeline:
    """
    RAG 全流程流水线: Load → Chunk → Embed → Index → Retrieve → Generate

    六步流水线说明:
      Load:     获取原始文档 (PDF/HTML → text)
      Chunk:    切分为合适大小的文本块
      Embed:    用 Sentence-Transformer 将文本转为向量
      Index:    存入 ChromaDB/FAISS 向量数据库 + 构建 BM25 索引
      Retrieve: 对查询执行混合检索 (BM25 + Vector → RRF 融合)
      Generate: 将检索到的文档 + 查询 → LLM 生成答案

    用法:
      pipeline = RAGPipeline(strategy="recursive_char")
      pipeline.build_index()                  # 构建索引
      result = pipeline.query("什么是 SVM?")   # 查询
    """

    def __init__(self, strategy: str = "recursive_char",
                 store_type: str = "chroma",
                 embedding_model: Optional[str] = None):
        self.strategy = strategy
        self.store_type = store_type
        self.embedding_model = embedding_model or config.default_embedding_model

        self.loader: Optional[DocumentLoader] = None
        self.chunker: Optional[Chunker] = None
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[Generator] = None
        self.chunks: List[Dict] = []

        self._init_components()

    def _init_components(self):
        self.loader = DocumentLoader()
        self.chunker = Chunker(strategy=self.strategy)
        self.embedder = Embedder(self.embedding_model)
        self.vector_store = VectorStore(store_type=self.store_type)
        self.retriever = HybridRetriever(self.embedder, self.vector_store)
        self.generator = Generator()

    def build_index(self, docs: Optional[List[Dict]] = None,
                    force: bool = False):
        """Load docs, chunk, embed, and build the search index.

        If the vector store already has data and force=False, skip
        embedding and reuse the existing index — only rebuild BM25.
        """
        if docs is None:
            docs = self.loader.run()

        if not docs:
            raise ValueError("No documents available. Check data source.")

        if not force and self.vector_store.count() > 0:
            self.chunks = self.chunker.chunk_documents(docs)
            self.retriever.build_bm25(self.chunks)
            self._log("使用已有索引 (%d 个向量)，跳过嵌入步骤",
                       self.vector_store.count())
            return

        self.chunks = self.chunker.chunk_documents(docs)
        if not self.chunks:
            raise ValueError("No chunks produced.")

        texts = [c["content"] for c in self.chunks]
        self._log("正在使用 %s 嵌入 %d 个文本块...",
                   self.embedding_model, len(texts))
        embeddings = self.embedder.embed(texts)

        self._log("正在构建 %s 索引，共 %d 个向量...",
                   self.store_type, len(embeddings))
        self.vector_store.add(self.chunks, embeddings)

        self.retriever.build_bm25(self.chunks)

        self._log("索引构建完成: %d 个向量, %d 个文本块 (来自 %d 篇文档)",
                   self.vector_store.count(), len(self.chunks), len(docs))

    @staticmethod
    def _log(fmt: str, *args):
        from logging import getLogger
        getLogger("rag_lab").info(fmt, *args)

    def query(self, question: str, top_k: int = 5,
              use_llm: bool = True) -> Dict:
        """Run a single query through the RAG pipeline."""
        t0 = time.time()

        retrieved = self.retriever.retrieve(question)
        if top_k != 5:
            retrieved = retrieved[:top_k]

        retrieval_time = time.time() - t0

        if use_llm:
            try:
                result = self.generator.generate(question, retrieved)
            except Exception as e:
                result = {
                    "query": question,
                    "answer": self.generator.fallback_response(question, retrieved),
                    "context_used": [d["content"][:200] for d in retrieved],
                    "sources": [d["metadata"]["source"] for d in retrieved],
                    "model": "fallback",
                }
        else:
            result = {
                "query": question,
                "answer": self.generator.fallback_response(question, retrieved),
                "context_used": [d["content"][:200] for d in retrieved],
                "sources": [d["metadata"]["source"] for d in retrieved],
                "model": "none",
            }

        result["retrieval_time_ms"] = round(retrieval_time * 1000)
        result["retrieved_docs"] = [
            {
                "id": d["id"],
                "source": d["metadata"]["source"],
                "score": round(d.get("score", 0), 4),
                "content_preview": d["content"][:150],
            }
            for d in retrieved
        ]
        return result

    def compare_retrieval_methods(self, question: str) -> Dict:
        """Compare BM25-only, vector-only, and hybrid retrieval."""
        bm25 = self.retriever.retrieve_bm25_only(question)
        vector = self.retriever.retrieve_vector_only(question)
        hybrid = self.retriever.retrieve(question)

        return {
            "bm25": self._format_retrieval(bm25),
            "vector": self._format_retrieval(vector),
            "hybrid": self._format_retrieval(hybrid),
            "translated_query": getattr(self.retriever, "_translated_query", None),
        }

    @staticmethod
    def _format_retrieval(results: List[Dict]) -> List[Dict]:
        return [
            {
                "id": d["id"],
                "source": d["metadata"]["source"],
                "score": round(d.get("score", 0), 4),
                "preview": d["content"][:120],
            }
            for d in results[:5]
        ]

    def rebuild_with_strategy(self, strategy: str, docs: List[Dict]):
        """Rebuild index with a different chunking strategy for comparison."""
        self.strategy = strategy
        self.chunker = Chunker(strategy=strategy)
        self.vector_store = VectorStore(store_type=self.store_type)
        self.retriever = HybridRetriever(self.embedder, self.vector_store)
        self.build_index(docs=docs)
