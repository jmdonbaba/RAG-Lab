import os
import re
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from config import config


class DocumentLoader:
    """Download and parse documents from CMU ML course lectures."""

    def __init__(self):
        self.raw_dir = config.raw_dir
        self.processed_dir = config.processed_dir
        config.ensure_dirs()

    def scrape_cmu_lectures(self) -> List[Dict]:
        """Scrape CMU 10-701 ML course lecture pages."""
        print(f"Scraping {config.cmu_lectures_url} ...")
        resp = requests.get(config.cmu_lectures_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # Find all lecture links
        lectures = []
        seen = set()
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") or href.endswith(".html"):
                full_url = href if href.startswith("http") else config.cmu_lecture_base + href
                title = link.get_text(strip=True)
                if full_url in seen:
                    continue
                seen.add(full_url)
                lectures.append({"title": title, "url": full_url})

        print(f"Found {len(lectures)} lecture files")
        return lectures

    def download_lectures(self, lectures: List[Dict]) -> List[str]:
        """Download lectures and save to raw directory."""
        saved = []
        for i, lec in enumerate(lectures):
            try:
                fname = f"lec_{i:02d}_{self._safe_name(lec['title'])}"
                ext = ".pdf" if lec["url"].endswith(".pdf") else ".html"
                fpath = os.path.join(self.raw_dir, fname + ext)

                if os.path.exists(fpath):
                    saved.append(fpath)
                    continue

                resp = requests.get(lec["url"], timeout=60)
                resp.raise_for_status()
                with open(fpath, "wb") as f:
                    f.write(resp.content)
                saved.append(fpath)
                print(f"  Downloaded: {fname}{ext}")
            except Exception as e:
                print(f"  Failed {lec['url']}: {e}")

        return saved

    def parse_html(self, filepath: str) -> str:
        """Parse HTML file to clean text."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        return self._clean_text(text)

    def parse_pdf(self, filepath: str) -> str:
        """Parse PDF file to text."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return self._clean_text("\n".join(texts))
        except ImportError:
            print("pypdf not installed, trying pdfplumber...")
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    texts = [p.extract_text() or "" for p in pdf.pages]
                return self._clean_text("\n".join(texts))
            except ImportError:
                raise ImportError("Need pypdf or pdfplumber to parse PDFs")

    def parse_documents(self, filepaths: List[str]) -> List[Dict]:
        """Parse all downloaded documents into structured format."""
        docs = []
        for fpath in filepaths:
            try:
                if fpath.endswith(".pdf"):
                    text = self.parse_pdf(fpath)
                else:
                    text = self.parse_html(fpath)

                if not text or len(text.strip()) < 100:
                    continue

                docs.append({
                    "source": os.path.basename(fpath),
                    "filepath": fpath,
                    "content": text,
                    "length": len(text),
                })
                print(f"  Parsed: {os.path.basename(fpath)} ({len(text)} chars)")
            except Exception as e:
                print(f"  Parse failed {fpath}: {e}")

        return docs

    def save_processed(self, docs: List[Dict]) -> str:
        """Save parsed documents as JSON."""
        output = []
        for d in docs:
            output.append({
                "source": d["source"],
                "content": d["content"],
                "length": d["length"],
            })

        outpath = os.path.join(self.processed_dir, "parsed_docs.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(output)} documents to {outpath}")
        return outpath

    def load_processed(self) -> List[Dict]:
        """Load previously processed documents."""
        inpath = os.path.join(self.processed_dir, "parsed_docs.json")
        if not os.path.exists(inpath):
            return []
        with open(inpath, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> List[Dict]:
        """Full pipeline: scrape -> download -> parse -> save."""
        cached = self.load_processed()
        if cached:
            print(f"Loaded {len(cached)} cached documents")
            return cached

        lectures = self.scrape_cmu_lectures()
        if not lectures:
            print("No lectures found, creating sample data...")
            return self._create_sample_docs()

        saved = self.download_lectures(lectures)
        if not saved:
            print("No files downloaded, creating sample data...")
            return self._create_sample_docs()

        docs = self.parse_documents(saved)
        if docs:
            self.save_processed(docs)
        return docs

    def _create_sample_docs(self) -> List[Dict]:
        """Create sample ML lecture documents for offline use."""
        samples = [
            {
                "title": "Linear Regression",
                "content": (
                    "Linear regression is a fundamental supervised learning algorithm that models "
                    "the relationship between a dependent variable y and one or more independent "
                    "variables X by fitting a linear equation. The model assumes y = w·x + b, where "
                    "w represents the weight vector and b is the bias term. The objective is to "
                    "minimize the mean squared error (MSE) between predicted and actual values: "
                    "L = (1/n) * Σ(y_i - ŷ_i)². This can be solved analytically via the normal "
                    "equation: w = (X^T X)^(-1) X^T y, or iteratively using gradient descent. "
                    "Ridge regression adds L2 regularization λ||w||² to prevent overfitting, "
                    "while Lasso (L1) regularization λ|w| enables feature selection by driving "
                    "some weights to exactly zero. The regularization strength λ is a hyperparameter "
                    "typically chosen via cross-validation."
                ),
            },
            {
                "title": "Logistic Regression",
                "content": (
                    "Logistic regression is a classification algorithm that estimates the probability "
                    "that an instance belongs to a particular class. Instead of fitting a straight line, "
                    "it uses the logistic (sigmoid) function σ(z) = 1/(1+e^(-z)) to map real-valued "
                    "outputs to [0,1] probabilities. The model learns weights by maximizing the "
                    "log-likelihood: L(w) = Σ[y_i log(σ(w·x_i)) + (1-y_i)log(1-σ(w·x_i))]. "
                    "Unlike linear regression, there is no closed-form solution, so we use iterative "
                    "optimization methods like gradient descent or Newton-Raphson. Logistic regression "
                    "can be extended to multi-class classification via one-vs-rest (OvR) or "
                    "multinomial (softmax) regression. The decision boundary is linear, making it "
                    "a linear classifier, but feature engineering and kernel methods can capture "
                    "non-linear patterns. Regularization (L1/L2) is crucial for high-dimensional data."
                ),
            },
            {
                "title": "Support Vector Machines",
                "content": (
                    "Support Vector Machines (SVMs) find the maximum-margin hyperplane that separates "
                    "classes in feature space. The margin is the distance from the decision boundary "
                    "to the nearest training samples (support vectors). SVM optimization solves: "
                    "min (1/2)||w||² subject to y_i(w·x_i + b) ≥ 1 for all i. For non-separable "
                    "data, soft-margin SVM introduces slack variables ξ_i and penalty C: "
                    "min (1/2)||w||² + C·Σξ_i. The kernel trick allows SVMs to learn non-linear "
                    "decision boundaries without explicitly computing feature transformations: "
                    "K(x_i, x_j) = φ(x_i)·φ(x_j). Common kernels include RBF (Gaussian), "
                    "polynomial, and sigmoid. SVMs are effective in high-dimensional spaces and "
                    "with clear margins of separation, though they scale poorly to very large datasets."
                ),
            },
            {
                "title": "Decision Trees and Ensemble Methods",
                "content": (
                    "Decision trees recursively partition the feature space based on simple rules. "
                    "At each node, the algorithm selects the feature and threshold that maximizes "
                    "information gain (reduction in entropy) or Gini impurity reduction. Entropy "
                    "H(S) = -Σ p_k log(p_k) measures node impurity. Decision trees are prone to "
                    "overfitting and high variance. Ensemble methods combine multiple trees to "
                    "improve robustness: Random Forests use bootstrap aggregating (bagging) to "
                    "train trees on random subsets and random feature subsets, then average "
                    "predictions. Gradient Boosting (XGBoost, LightGBM, CatBoost) builds trees "
                    "sequentially where each new tree corrects the errors of previous trees by "
                    "fitting to the negative gradient of the loss function. These are among the "
                    "most powerful algorithms for tabular data."
                ),
            },
            {
                "title": "Neural Networks and Deep Learning",
                "content": (
                    "Neural networks are composed of layers of interconnected neurons. Each neuron "
                    "computes a weighted sum of inputs plus bias, passed through a nonlinear "
                    "activation function: a = f(W·x + b). Common activations: ReLU(x)=max(0,x), "
                    "sigmoid, tanh, and softmax for output probabilities. Multi-layer perceptrons "
                    "(MLPs) with at least one hidden layer are universal function approximators. "
                    "Training uses backpropagation: compute gradients of the loss with respect to "
                    "each parameter via the chain rule, then update with gradient descent: "
                    "θ = θ - η·∇L. Modern deep learning uses automatic differentiation and GPU "
                    "acceleration. Key architectures: CNNs for spatial data (convolutions, pooling), "
                    "RNNs/LSTMs/GRUs for sequential data, Transformers with self-attention for "
                    "NLP and beyond. Regularization techniques include dropout, batch normalization, "
                    "weight decay, and early stopping."
                ),
            },
            {
                "title": "Bias-Variance Tradeoff",
                "content": (
                    "The bias-variance tradeoff is a fundamental concept in statistical learning "
                    "theory. The expected prediction error can be decomposed as: "
                    "Error = Bias² + Variance + Irreducible Error. Bias measures how far the "
                    "average prediction is from the true value (systematic error from model "
                    "assumptions). Variance measures how much predictions vary across different "
                    "training sets (sensitivity to data). Simple models (linear regression) have "
                    "high bias and low variance; complex models (deep neural nets) have low bias "
                    "but high variance. The goal is to find the sweet spot that minimizes total "
                    "error. Regularization, cross-validation, and ensemble methods help manage "
                    "this tradeoff. In the modern deep learning era with massive datasets, "
                    "very large models can achieve both low bias and low variance, challenging "
                    "the classical U-shaped curve."
                ),
            },
            {
                "title": "Gradient Descent and Optimization",
                "content": (
                    "Gradient descent is the workhorse optimization algorithm for machine learning. "
                    "It iteratively updates parameters in the direction of steepest descent of the "
                    "loss function: θ_{t+1} = θ_t - η·∇L(θ_t). The learning rate η controls step "
                    "size — too large causes divergence, too small causes slow convergence. "
                    "Variants include: Batch GD (full dataset per step, expensive), Stochastic GD "
                    "(one sample per step, noisy but fast), Mini-batch GD (compromise). Momentum "
                    "accelerates convergence by accumulating a velocity vector: v = βv + (1-β)∇L. "
                    "Adam combines momentum with adaptive learning rates per parameter using first "
                    "and second moment estimates: m_t = β₁m_{t-1} + (1-β₁)g_t, v_t = β₂v_{t-1} + "
                    "(1-β₂)g_t². AdamW adds decoupled weight decay for better regularization. "
                    "Learning rate schedules (cosine, step, warmup) are crucial for convergence."
                ),
            },
            {
                "title": "Cross-Validation and Model Selection",
                "content": (
                    "Cross-validation is the standard method for estimating model performance and "
                    "selecting hyperparameters. K-fold CV partitions data into K equal folds, "
                    "trains on K-1 folds and validates on the remaining fold, repeating K times. "
                    "The average validation score provides an unbiased estimate of generalization "
                    "error. Stratified K-fold preserves class proportions in each fold. Leave-one-out "
                    "CV (LOOCV) uses N-1 training samples per fold — low bias but high variance "
                    "and computational cost. Nested CV uses an outer loop for performance estimation "
                    "and an inner loop for hyperparameter tuning, preventing information leakage. "
                    "For time series, use time-based splits (not random) to respect temporal order. "
                    "Common pitfall: using CV to both tune hyperparameters AND report performance "
                    "leads to optimistic bias."
                ),
            },
            {
                "title": "EM Algorithm",
                "content": (
                    "The Expectation-Maximization (EM) algorithm estimates parameters in models "
                    "with latent (unobserved) variables. It alternates between two steps: "
                    "E-step: Compute the expected complete-data log-likelihood given current "
                    "parameters and observed data: Q(θ|θ^t) = E[log P(X,Z|θ) | X, θ^t]. "
                    "M-step: Maximize Q with respect to θ to obtain new parameters: "
                    "θ^{t+1} = argmax_θ Q(θ|θ^t). EM guarantees that the observed data "
                    "log-likelihood L(θ) = log P(X|θ) never decreases. Classic applications "
                    "include: Gaussian Mixture Models (GMMs) for clustering, where latent "
                    "variables indicate cluster assignments; Hidden Markov Models (HMMs) for "
                    "sequential data; and collaborative filtering with missing values. "
                    "EM converges to a local maximum and is sensitive to initialization."
                ),
            },
            {
                "title": "Bayesian Inference",
                "content": (
                    "Bayesian inference treats model parameters as random variables with prior "
                    "distributions, updating beliefs based on observed data via Bayes' theorem: "
                    "P(θ|D) = P(D|θ)·P(θ) / P(D). The posterior P(θ|D) combines prior knowledge "
                    "P(θ) with the likelihood P(D|θ). The evidence P(D) = ∫P(D|θ)P(θ)dθ is the "
                    "marginal likelihood, often intractable. Maximum a Posteriori (MAP) estimation "
                    "finds the mode of the posterior; full Bayesian inference marginalizes over "
                    "parameters. Conjugate priors (e.g., Beta-Binomial, Gaussian-Gaussian) yield "
                    "closed-form posteriors. For complex models, approximate inference methods "
                    "include Markov Chain Monte Carlo (MCMC) sampling and Variational Inference "
                    "(VI), which turns inference into optimization. Bayesian methods naturally "
                    "provide uncertainty estimates and resist overfitting through prior regularization."
                ),
            },
        ]

        docs = []
        for s in samples:
            docs.append({
                "source": s["title"].replace(" ", "_") + ".txt",
                "filepath": os.path.join(self.raw_dir, s["title"].replace(" ", "_") + ".txt"),
                "content": s["content"],
                "length": len(s["content"]),
            })
        self.save_processed(docs)
        return docs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
