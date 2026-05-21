import sys
import os
import tempfile
import shutil

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------- Sample test data --------------------------------------------------

SAMPLE_ML_EN = (
    "Linear regression is a fundamental supervised learning algorithm. "
    "It models the relationship between a dependent variable and independent variables.\n\n"
    "Gradient descent is the workhorse optimization algorithm. "
    "It iteratively updates parameters in the direction of steepest descent. "
    "The learning rate controls the step size.\n\n"
    "Support vector machines find the maximum-margin hyperplane. "
    "The kernel trick allows SVMs to learn non-linear decision boundaries. "
    "Common kernels include RBF, polynomial, and sigmoid."
)

SAMPLE_ML_CN = (
    "逻辑回归是一种分类算法，它估计样本属于某个类别的概率。\n\n"
    "梯度下降是机器学习中最常用的优化算法。它通过迭代更新参数来最小化损失函数。"
)

SAMPLE_ML_DOCS = [
    {
        "source": "linear_regression.txt",
        "filepath": "/tmp/linear_regression.txt",
        "content": (
            "Linear regression is a fundamental supervised learning algorithm that models "
            "the relationship between a dependent variable y and independent variables X. "
            "The model fits a linear equation y = wx + b to observed data. "
            "The objective is to minimize the mean squared error between predicted and actual values. "
            "Ridge regression adds L2 regularization to prevent overfitting, "
            "while Lasso regression uses L1 regularization for feature selection."
        ),
        "length": 420,
    },
    {
        "source": "logistic_regression.txt",
        "filepath": "/tmp/logistic_regression.txt",
        "content": (
            "Logistic regression is a classification algorithm that estimates class probabilities. "
            "It uses the sigmoid function sigma(z) = 1/(1+e^(-z)) to map outputs to [0, 1]. "
            "The model is trained by maximizing the log-likelihood of the observed data. "
            "Unlike linear regression, there is no closed-form solution, "
            "so iterative optimization methods like gradient descent are used. "
            "Regularization techniques such as L1 and L2 penalties help prevent overfitting."
        ),
        "length": 430,
    },
    {
        "source": "gradient_descent.txt",
        "filepath": "/tmp/gradient_descent.txt",
        "content": (
            "Gradient descent is the workhorse optimization algorithm for machine learning. "
            "It iteratively updates parameters in the direction of steepest descent. "
            "The update rule is theta = theta - learning_rate * gradient. "
            "Stochastic gradient descent uses a single sample per update, "
            "while batch gradient descent uses the entire dataset. "
            "Momentum helps accelerate convergence by accumulating past gradients."
        ),
        "length": 410,
    },
    {
        "source": "svm.txt",
        "filepath": "/tmp/svm.txt",
        "content": (
            "Support Vector Machines find the optimal separating hyperplane between classes. "
            "The margin is the distance from the decision boundary to the nearest data points. "
            "Soft-margin SVMs use slack variables to handle non-separable data. "
            "The kernel trick K(x_i, x_j) enables non-linear classification. "
            "Common kernels include RBF (Gaussian), polynomial, and sigmoid. "
            "SVMs are particularly effective in high-dimensional spaces."
        ),
        "length": 430,
    },
]


@pytest.fixture
def sample_en_text():
    return SAMPLE_ML_EN


@pytest.fixture
def sample_cn_text():
    return SAMPLE_ML_CN


@pytest.fixture
def sample_docs():
    return [dict(d) for d in SAMPLE_ML_DOCS]


# ------- Chunker fixtures --------------------------------------------------

@pytest.fixture
def sample_chunks(sample_docs):
    from src.chunker import Chunker
    chunker = Chunker(strategy="recursive_char", chunk_size=256, chunk_overlap=30)
    return chunker.chunk_documents(sample_docs)


# ------- Embedder (session-scoped, loaded once) -----------------------------

@pytest.fixture(scope="session")
def embedder():
    """Session-scoped embedder using the lightest model for speed.

    Forces offline mode so it loads instantly from the HuggingFace cache
    instead of phoning home (especially important behind China's GFW).
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from src.embedder import Embedder
        emb = Embedder(model_name="all-MiniLM-L6-v2")
        _ = emb.embed(["warmup"], show_progress=False)
        return emb
    except Exception as exc:
        pytest.skip(f"Embedding model unavailable: {exc}")


# ------- Vector store temp dirs ---------------------------------------------

@pytest.fixture
def chroma_tmp_path(tmp_path):
    """Temporary directory for ChromaDB so real data/ is untouched."""
    chroma_dir = tmp_path / "chroma_test_db"
    chroma_dir.mkdir()
    return str(chroma_dir)


# ------- Small embeddings (for tests that need pre-computed vectors) --------

@pytest.fixture
def sample_embeddings(embedder):
    """32-dim random embeddings matching a hypothetical model dimension."""
    texts = [d["content"] for d in SAMPLE_ML_DOCS]
    return embedder.embed(texts, show_progress=False)
