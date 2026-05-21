"""Generate figures for RAG experiment report from actual evaluation data.

Usage:  python scripts/generate_figures.py
Output: figures/*.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Force offline mode so models load from cache instantly
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

STRATEGY_LABELS = {
    "fixed_token": "Fixed Token",
    "recursive_char": "Recursive Char",
    "semantic": "Semantic",
}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    from config import config
    from src.pipeline import RAGPipeline
    from src.evaluator import Evaluator

    print("Building pipeline (offline mode)...")
    pipeline = RAGPipeline()
    pipeline.build_index()

    evaluator = Evaluator(pipeline)
    print("Running full evaluation (may take 1-2 minutes)...")
    results = evaluator.run_full_evaluation()

    fig_chunking(results)
    fig_retrieval(results)
    fig_embedding(results)
    fig_coverage(results, pipeline)
    fig_retrieval_methods(pipeline)

    print(f"\nDone — {len(os.listdir(FIG_DIR))} figures saved to {FIG_DIR}/")


# ---- Figure 1: Chunking Strategy Comparison ---------------------------------


def fig_chunking(results):
    data = results.get("chunking_comparison", {})
    if not data:
        print("  SKIP chunking — no data")
        return

    strategies = list(data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    metrics = [
        ("num_chunks", "Number of Chunks", "Chunks"),
        ("avg_chunk_length", "Average Chunk Length", "Characters"),
        ("avg_latency_ms", "Avg Retrieval Latency", "Latency (ms)"),
    ]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [data[s][key] for s in strategies]
        bars = ax.bar(strategies, values, color=COLORS[:3], width=0.5, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(
            [STRATEGY_LABELS.get(s, s) for s in strategies], rotation=15, ha="right"
        )
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val:.0f}", ha="center", fontsize=9)

    fig.suptitle("Chunking Strategy Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "chunking_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


# ---- Figure 2: Retrieval Performance per Query ------------------------------


def fig_retrieval(results):
    retrieval = results.get("retrieval", {})
    per_query = retrieval.get("per_query", [])
    if not per_query:
        print("  SKIP retrieval performance — no data")
        return

    overlaps = [q["bm25_vector_jaccard"] for q in per_query]
    latencies = [q["latency_ms"] for q in per_query]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = range(len(per_query))
    labels = [f"Q{i + 1}" for i in x]

    axes[0].bar(x, overlaps, color=COLORS[1], width=0.5, edgecolor="white")
    axes[0].set_title("BM25-Vector Overlap (Jaccard)")
    axes[0].set_ylabel("Jaccard Coefficient")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].axhline(y=np.mean(overlaps), color=COLORS[3], linestyle="--",
                    label=f"Avg: {np.mean(overlaps):.3f}")
    axes[0].legend(fontsize=9)

    axes[1].bar(x, latencies, color=COLORS[2], width=0.5, edgecolor="white")
    axes[1].set_title("Retrieval Latency per Query")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(y=retrieval["avg_latency_ms"], color=COLORS[3], linestyle="--",
                    label=f"Avg: {retrieval['avg_latency_ms']}ms")
    axes[1].legend(fontsize=9)

    fig.suptitle("Retrieval Performance Across Evaluation Queries", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "retrieval_performance.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


# ---- Figure 3: Embedding Model Comparison -----------------------------------


def fig_embedding(results):
    data = results.get("embedding_comparison", {})
    if not data:
        print("  SKIP embedding comparison — no data")
        return

    models = list(data.keys())
    short = [m.split("/")[-1][:25] for m in models]

    similarities = [data[m]["avg_similarity"] for m in models]
    coverages = [data[m]["coverage"] for m in models]
    dims = [data[m]["dim"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(short, similarities, color=COLORS[: len(models)], width=0.5, edgecolor="white")
    axes[0].set_title("Average Cosine Similarity")
    axes[0].set_ylabel("Similarity")
    axes[0].set_xticklabels(short, rotation=15, ha="right")
    axes[0].set_ylim(0, max(similarities) * 1.3)

    axes[1].bar(short, coverages, color=COLORS[: len(models)], width=0.5, edgecolor="white")
    axes[1].set_title("Coverage (similarity > 0.3)")
    axes[1].set_ylabel("Fraction")
    axes[1].set_xticklabels(short, rotation=15, ha="right")
    axes[1].set_ylim(0, 1.1)

    axes[2].bar(short, dims, color=COLORS[: len(models)], width=0.5, edgecolor="white")
    axes[2].set_title("Embedding Dimension")
    axes[2].set_ylabel("Dimensions")
    axes[2].set_xticklabels(short, rotation=15, ha="right")
    for i, d in enumerate(dims):
        axes[2].text(i, d + 10, str(d), ha="center", fontsize=10)

    fig.suptitle("Embedding Model Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "embedding_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


# ---- Figure 4: Hybrid retrieval quality per query ---------------------------


def fig_coverage(results, pipeline):
    retrieval = results.get("retrieval", {})
    per_query = retrieval.get("per_query", [])
    if not per_query:
        print("  SKIP coverage — no data")
        return

    top_scores = [q["hybrid_top_score"] for q in per_query]

    # Collect average top-5 score per query from actual retrieval results
    avg_scores = []
    for q_info in per_query:
        retrieved = pipeline.retriever.retrieve(q_info["query"])
        s = [r.get("score", 0) for r in retrieved[:5]]
        avg_scores.append(np.mean(s) if s else 0)

    x = range(len(per_query))
    labels = [f"Q{i + 1}" for i in x]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    axes[0].bar(x, top_scores, color=COLORS[0], width=0.5, edgecolor="white")
    axes[0].set_title("Hybrid Top-1 Score per Query")
    axes[0].set_ylabel("RRF Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)

    axes[1].bar(x, avg_scores, color=COLORS[4], width=0.5, edgecolor="white")
    axes[1].set_title("Hybrid Avg Score (Top-5)")
    axes[1].set_ylabel("Avg RRF Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(y=np.mean(avg_scores), color=COLORS[2], linestyle="--",
                    label=f"Avg: {np.mean(avg_scores):.2f}")
    axes[1].legend(fontsize=9)

    fig.suptitle("Hybrid Retrieval Quality — Best vs Average Score",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "hybrid_quality.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


# ---- Figure 5: Retrieval Method Comparison (pairwise Jaccard) ----------------


def fig_retrieval_methods(pipeline):
    queries = [
        "What is logistic regression?",
        "Explain gradient descent",
        "What is Bayesian inference?",
        "How do decision trees work?",
        "Explain the EM algorithm",
    ]

    pairs = [("bm25", "vector"), ("bm25", "hybrid"), ("vector", "hybrid")]
    pair_labels = {"bm25-vector": "BM25 vs Vector",
                   "bm25-hybrid": "BM25 vs Hybrid",
                   "vector-hybrid": "Vector vs Hybrid"}

    overlaps = {f"{a}-{b}": [] for a, b in pairs}
    for q in queries:
        results = pipeline.compare_retrieval_methods(q)
        for a, b in pairs:
            ids_a = {d["id"] for d in results.get(a, []) if d.get("id")}
            ids_b = {d["id"] for d in results.get(b, []) if d.get("id")}
            union = len(ids_a | ids_b)
            inter = len(ids_a & ids_b)
            overlaps[f"{a}-{b}"].append(inter / max(union, 1))

    x = np.arange(len(queries))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (a, b) in enumerate(pairs):
        key = f"{a}-{b}"
        ax.bar(x + i * width, overlaps[key], width, label=pair_labels[key],
               color=COLORS[i], edgecolor="white")

    ax.set_title("Retrieval Method Overlap (Jaccard)", fontweight="bold")
    ax.set_ylabel("Jaccard Coefficient")
    ax.set_xticks(x + width)
    ax.set_xticklabels([q[:35] + "..." for q in queries], rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "retrieval_methods.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
