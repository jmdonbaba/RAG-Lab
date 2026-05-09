import time
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

    def evaluate_retrieval(self, queries: Optional[List[str]] = None,
                           metrics: Optional[List[str]] = None) -> Dict:
        """Evaluate retrieval quality: MRR, Recall@K, NDCG@K."""
        if queries is None:
            queries = config.eval_queries
        if metrics is None:
            metrics = ["recall@3", "recall@5", "mrr", "ndcg@5"]

        # For retrieval evaluation without ground truth, measure:
        # - self-consistency (BM25 vs vector overlap)
        # - score distribution
        # - retrieval latency
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

            # Overlap between methods
            bm25_vector_overlap.append(
                len(bm25_ids & vector_ids) / max(len(bm25_ids | vector_ids), 1)
            )

            # Score statistics
            scores = [r.get("score", 0) for r in hybrid]
            if scores:
                score_variance.append(float(np.var(scores)))

            all_results.append({
                "query": q,
                "num_hybrid": len(hybrid),
                "num_bm25": len(bm25),
                "num_vector": len(vector),
                "bm25_vector_jaccard": bm25_vector_overlap[-1],
                "hybrid_top_score": scores[0] if scores else 0,
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
        """Compare retrieval under different chunking strategies."""
        if queries is None:
            queries = config.eval_queries[:5]

        comparison = {}
        docs = self.pipeline.loader.load_processed()
        if not docs:
            docs = self.pipeline.loader.run()

        for strategy in ["fixed_token", "recursive_char", "semantic"]:
            print(f"\n  Evaluating strategy: {strategy}")
            self.pipeline.rebuild_with_strategy(strategy, docs)
            eval_result = self.evaluate_retrieval(queries)
            comparison[strategy] = {
                "num_chunks": len(self.pipeline.chunks),
                "avg_chunk_length": np.mean([c["length"] for c in self.pipeline.chunks])
                if self.pipeline.chunks else 0,
                "avg_latency_ms": eval_result["avg_latency_ms"],
                "avg_bm25_vector_overlap": eval_result["avg_bm25_vector_overlap"],
            }

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
        comparison = compare_embedding_models(
            texts, "What is logistic regression?"
        )
        self.results["embedding_comparison"] = comparison
        return comparison

    def evaluate_generation_quality(self, queries: Optional[List[str]] = None) -> Dict:
        """Evaluate generation quality (uses ground truth from provided context)."""
        if queries is None:
            queries = config.eval_queries[:3]

        gen_results = []
        for q in queries:
            try:
                result = self.pipeline.query(q, use_llm=False)
            except Exception as e:
                result = {"answer": str(e), "retrieved_docs": []}

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
        print("=" * 60)
        print("RAG SYSTEM EVALUATION")
        print("=" * 60)

        print("\n[1/4] Evaluating retrieval quality...")
        self.evaluate_retrieval()
        print(f"  Avg latency: {self.results['retrieval']['avg_latency_ms']}ms")
        print(f"  Avg BM25-Vector overlap: {self.results['retrieval']['avg_bm25_vector_overlap']}")

        print("\n[2/4] Comparing chunking strategies...")
        self.evaluate_chunking_strategies()
        for s, r in self.results["chunking_comparison"].items():
            print(f"  {s}: {r['num_chunks']} chunks, latency={r['avg_latency_ms']}ms")

        print("\n[3/4] Comparing embedding models...")
        self.evaluate_embedding_models()

        print("\n[4/4] Evaluating generation quality...")
        self.evaluate_generation_quality()

        return self.results

    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            self.run_full_evaluation()

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        if "retrieval" in self.results:
            r = self.results["retrieval"]
            print(f"\n--- Retrieval Performance ---")
            print(f"  Queries evaluated: {r['num_queries']}")
            print(f"  Avg latency: {r['avg_latency_ms']}ms")
            print(f"  Avg BM25-Vector overlap: {r['avg_bm25_vector_overlap']}")

        if "chunking_comparison" in self.results:
            print(f"\n--- Chunking Strategy Comparison ---")
            for name, data in self.results["chunking_comparison"].items():
                print(f"  {name:20s}: chunks={data['num_chunks']:4d}, "
                      f"avg_len={data['avg_chunk_length']:.0f}, "
                      f"latency={data['avg_latency_ms']}ms")

        if "embedding_comparison" in self.results:
            print(f"\n--- Embedding Model Comparison ---")
            for model, data in self.results["embedding_comparison"].items():
                print(f"  {model}: dim={data['dim']}, "
                      f"avg_sim={data['avg_similarity']:.4f}")

        if "generation" in self.results:
            g = self.results["generation"]
            print(f"\n--- Generation Quality ---")
            print(f"  Avg answer length: {g['avg_answer_length']:.0f} chars")
            print(f"  Avg sources per query: {g['avg_sources_used']:.1f}")
