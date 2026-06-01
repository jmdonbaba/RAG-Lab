"""
RAG-Lab 全局配置

所有可调参数集中在这里。教程建议的修改顺序:
  1. 先改 chunk_size / chunk_overlap 观察检索结果变化
  2. 再换 embedding_models 对比语义理解差异
  3. 调 retrieval_weights 看关键词 vs 语义的权衡
  4. 最后换 llm_provider 对比生成质量
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 镜像 — 国内网络环境下自动切换，解决 HuggingFace Hub 被墙问题
# 参考: https://hf-mirror.com
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 项目级 logger，所有模块通过 logging.getLogger("rag_lab") 引用
logger = logging.getLogger("rag_lab")


@dataclass
class Config:
    # ============================================================
    # 路径配置
    # ============================================================
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(project_root, "data")
    raw_dir: str = os.path.join(data_dir, "raw")
    processed_dir: str = os.path.join(data_dir, "processed")
    chroma_dir: str = os.path.join(data_dir, "chroma_db")
    faiss_dir: str = os.path.join(data_dir, "faiss_index")

    # ============================================================
    # 知识源 — CMU 10-701 机器学习课程
    # 替换此处可接入你自己的文档
    # ============================================================
    cmu_lectures_url: str = "https://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml"
    cmu_lecture_base: str = "https://www.cs.cmu.edu/~tom/10701_sp11/"

    # ============================================================
    # 分块 (Chunking)
    #
    # chunk_size: 每个文本块的目标长度（字符数）
    #   - 太小: 语义碎片化，检索可能丢失上下文
    #   - 太大: embedding 模型可能截断，检索精度下降
    #   - 建议范围: 256-1024，512 是常用起点
    #
    # chunk_overlap: 相邻块之间的重叠字符数
    #   - 防止关键信息落在分块边界
    #   - 通常设为 chunk_size 的 10-20%
    # ============================================================
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunk_strategies: List[str] = field(
        default_factory=lambda: ["fixed_token", "recursive_char", "semantic"]
    )

    # ============================================================
    # 嵌入模型 (Embedding)
    #
    # 首次使用会自动下载模型权重:
    #   - all-MiniLM-L6-v2:   英文轻量，384维，~80MB
    #   - bge-small-zh-v1.5:  BGE 中文，512维，~100MB
    #   - moka-ai/m3e-base:   M3E 中文，768维，~400MB
    #
    # 切换模型后需要重新构建索引 (python main.py build)
    # ============================================================
    embedding_models: List[str] = field(
        default_factory=lambda: [
            "all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
            "moka-ai/m3e-base",
        ]
    )
    default_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ============================================================
    # 向量存储 (Vector Store)
    #
    # chroma: 嵌入式向量数据库，自带持久化 (SQLite)
    # faiss:  Meta 高性能库，纯内存，速度快但不持久化
    # ============================================================
    vector_store_type: str = "chroma"
    collection_name: str = "cmu_ml_lectures"

    # ============================================================
    # 检索 (Retrieval)
    #
    # top_k: 每次返回的文档块数量
    # retrieval_weights: BM25 和向量检索的权重
    #   - bm25=0.2, vector=0.8: 偏语义
    #   - bm25=0.4, vector=0.6: 推荐（兼顾关键词与语义）
    #   - bm25=0.5, vector=0.5: 均衡
    #   - bm25=0.8, vector=0.2: 偏关键词
    #
    # hybrid_fusion: 融合算法
    #   - "rrf":      Reciprocal Rank Fusion（基于排名，推荐）
    #   - "weighted": 归一化后加权求和
    # ============================================================
    top_k: int = 5
    retrieval_weights: dict = field(default_factory=lambda: {"bm25": 0.4, "vector": 0.6})
    hybrid_fusion: str = "rrf"
    enable_query_translation: bool = True  # BM25 中→英翻译，解决跨语言检索

    # ============================================================
    # 答案生成 (Generation)
    #
    # deepseek: 性价比高，1M上下文，中英文俱佳 (推荐)
    # openai:   GPT系列，质量稳定
    # local:    本地模型 (需要 GPU，需自行安装 transformers)
    #
    # temperature: 0.0-2.0，越低越确定/保守
    #   - 知识问答建议 0.1（减少编造）
    #   - 创意生成建议 0.7+
    # ============================================================
    llm_provider: str = "deepseek"
    deepseek_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = "gpt-3.5-turbo"
    local_model: str = "Qwen/Qwen2-7B-Instruct"
    max_context_length: int = 1024 * 1024
    temperature: float = 0.1

    # ============================================================
    # 评估 (Evaluation)
    #
    # 内置 10 条评估查询，覆盖机器学习核心概念。
    # 可替换为你自己领域的评估查询。
    # ============================================================
    eval_queries: List[str] = field(
        default_factory=lambda: [
            "What is logistic regression?",
            "Explain the concept of support vector machines",
            "What is the bias-variance tradeoff?",
            "How does gradient descent work?",
            "What is cross-validation?",
            "Explain the EM algorithm",
            "What is Bayesian inference?",
            "How do decision trees work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain principal component analysis",
        ]
    )

    def ensure_dirs(self):
        """创建所有必要的数据目录."""
        for d in [self.data_dir, self.raw_dir, self.processed_dir,
                  self.chroma_dir, self.faiss_dir]:
            os.makedirs(d, exist_ok=True)


# 全局单例，所有模块通过 `from config import config` 引用
config = Config()
