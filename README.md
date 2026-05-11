<div align="center">

# 🔬 RAG-Lab

**Retrieval-Augmented Generation — 从原理到实现的动手实验**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://www.gradio.app/)

</div>

---

## 为什么需要 RAG？

大语言模型（LLM）虽然强大，但存在两个根本问题：

- **知识截止日期**：训练数据有时效性，无法回答最新问题
- **幻觉（Hallucination）**：模型可能编造不存在的事实

RAG 的核心思路是 **先检索，再生成** —— 从外部知识库中检索相关文档，将其作为上下文喂给 LLM，让模型基于真实资料回答问题。

```
用户提问 → 向量检索 → 找到相关文档 → 拼接上下文 → LLM 生成回答
```

---

## 项目结构

```
RAG-Lab/
├── README.md
├── requirements.txt
├── .env.example                  # API Key 配置模板
├── config.py                     # 全局配置（数据路径、模型、参数）
├── main.py                       # CLI 命令行入口
│
├── notebooks/                    # 📓 Jupyter 教程（推荐从这里开始）
│   ├── 01_document_loading.ipynb
│   ├── 02_embedding_indexing.ipynb
│   ├── 03_retrieval_generation.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                          # 核心源码
│   ├── document_loader.py        # 文档抓取、下载、解析
│   ├── chunker.py                # 文本分块策略（固定Token/递归字符/语义）
│   ├── embedder.py               # Sentence-Transformer 向量嵌入
│   ├── vector_store.py           # ChromaDB / FAISS 向量存储
│   ├── retriever.py              # BM25 + 向量混合检索（RRF融合）
│   ├── generator.py              # LLM 答案生成（DeepSeek/OpenAI/本地）
│   ├── pipeline.py               # 完整 RAG 流水线
│   ├── evaluator.py              # 检索与生成质量评估
│   └── app.py                    # Gradio 网页界面
│
└── data/                         # 数据目录（自动创建，已 gitignore）
    ├── raw/                      # 原始下载文件
    ├── processed/                # 解析后的文本
    ├── chroma_db/                # ChromaDB 持久化索引
    └── faiss_index/              # FAISS 索引文件
```

---

## 数据集

本项目使用 **CMU 10-701 机器学习课程**（2011 年春季学期）的公开讲义作为知识库。

| 项目 | 信息 |
|------|------|
| 课程 | 10-701 Introduction to Machine Learning |
| 教授 | Tom Mitchell |
| 学期 | Spring 2011 |
| 来源 | https://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml |
| 文件数 | 43 篇 PDF 讲义 |
| 许可 | 公开课程资料，仅用于学习研究 |

**涵盖主题**：线性回归、逻辑回归、朴素贝叶斯、SVM、决策树、神经网络、深度学习、偏差-方差权衡、梯度下降、交叉验证、EM 算法、贝叶斯推断、图模型、PAC 学习、PCA/SVD、强化学习 等。

首次运行 `python main.py build` 时会自动从 CMU 网站抓取并下载所有讲义 PDF，解析为纯文本后构建索引。离线环境下会自动使用内置的 10 篇示例文档。

> 如果想接入你自己的文档，只需修改 `config.py` 中的 `cmu_lectures_url` 和 `cmu_lecture_base` 即可。

---

## 快速开始

### 1. 环境准备

```bash
git clone <your-repo-url>
cd RAG-Lab

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 配置 LLM API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

| 提供商 | 获取地址 | 说明 |
|--------|---------|------|
| DeepSeek（推荐） | https://platform.deepseek.com/api_keys | 性价比高，1M 上下文 |
| OpenAI | https://platform.openai.com/api-keys | GPT 系列，质量稳定 |

不想调用 LLM？使用 `--no_llm` 模式，仅展示检索结果。

### 3. 首次运行

```bash
# 构建索引（自动下载 CMU 课程讲义 → 分块 → 嵌入 → 存储）
python main.py build

# 提问
python main.py query "What is logistic regression?"

# 对比不同检索策略
python main.py compare "Explain gradient descent"

# 启动网页界面
python main.py ui
```

---

## CLI 命令一览

| 命令 | 说明 | 示例 |
|------|------|------|
| `build` | 构建搜索索引 | `python main.py build --strategy recursive_char --store chroma` |
| `query` | 单次问答 | `python main.py query "什么是EM算法？" --top_k 5` |
| `compare` | 对比检索策略 | `python main.py compare "贝叶斯推断"` |
| `evaluate` | 运行完整评估 | `python main.py evaluate` |
| `ui` | 启动 Gradio 界面 | `python main.py ui` |

### 常用参数

| 参数 | 命令 | 说明 |
|------|------|------|
| `--strategy` | `build` | 分块策略：`fixed_token` / `recursive_char` / `semantic` |
| `--store` | `build` | 向量存储：`chroma`（持久化）/ `faiss`（纯内存） |
| `--embedding` | `build` | 嵌入模型名称 |
| `--top_k` | `query` | 返回的文档片段数量（默认 5） |
| `--no_llm` | `query` | 仅检索模式，不调用 LLM 生成 |

---

## 教程路线

### 第1课：文档加载与解析
[`notebooks/01_document_loading.ipynb`](notebooks/01_document_loading.ipynb)

- 从网页抓取文档列表 → 下载 PDF/HTML → 解析提取纯文本 → 清洗存储

**核心概念**：RAG 的第一步是获取知识。本课展示如何将任意文档转化为可检索的文本。

### 第2课：向量嵌入与索引构建
[`notebooks/02_embedding_indexing.ipynb`](notebooks/02_embedding_indexing.ipynb)

- 三种分块策略对比（固定Token / 递归字符 / 语义）→ Sentence-Transformer 向量化 → ChromaDB & FAISS 存储

**核心概念**：检索质量取决于"怎么切"和"怎么存"。分块太大检索不准，太小丢失上下文。

### 第3课：混合检索与答案生成
[`notebooks/03_retrieval_generation.ipynb`](notebooks/03_retrieval_generation.ipynb)

- BM25 关键词检索 + 向量语义检索 → RRF 融合 → Prompt 模板设计 → LLM 生成 + 来源追溯

**核心概念**：单一检索方式各有盲区——BM25 不懂语义，向量检索可能漏掉关键词。混合检索取长补短，是生产环境的标配。

### 第4课：系统评估与策略对比
[`notebooks/04_evaluation.ipynb`](notebooks/04_evaluation.ipynb)

- 检索质量评估（延迟、召回、重叠度）→ 分块策略对比 → 嵌入模型对比 → 端到端评估报告

**核心概念**：没有评估就没有改进方向。本课量化对比不同配置下的 RAG 系统表现。

---

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 文档解析 | `pypdf` + `BeautifulSoup` |
| 文本分块 | `tiktoken`（Token级）+ 递归/语义分块 |
| 嵌入模型 | `sentence-transformers`（BGE / M3E / MiniLM） |
| 向量存储 | ChromaDB（默认）/ FAISS |
| 关键词检索 | BM25（`rank-bm25`） |
| 答案生成 | DeepSeek API / OpenAI API / 本地模型 |
| 评估框架 | 自建（延迟、重叠度、分块对比） |
| 界面 | Gradio 4.x |

---

## 配置说明

编辑 `config.py` 进行自定义配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 | 文本块最大字符数（建议 256-1024） |
| `chunk_overlap` | 50 | 相邻块重叠字符数（建议 chunk_size 的 10-20%） |
| `top_k` | 5 | 每次检索返回文档数 |
| `retrieval_weights` | BM25 0.2 / Vector 0.8 | 混合检索权重配比 |
| `hybrid_fusion` | `"rrf"` | 融合算法：`rrf`（排名融合）/ `weighted`（加权求和） |
| `llm_provider` | `"deepseek"` | LLM 提供商：`deepseek` / `openai` / `local` |
| `temperature` | 0.1 | 生成温度（越低越保守，适合知识问答） |
| `enable_query_translation` | `True` | BM25 中→英查询翻译（跨语言检索） |

---

## 功能亮点

### 🌐 BM25 查询翻译

文档为英文讲义，但用户可能用中文提问。系统自动检测中文查询，通过 LLM 翻译为英文后再执行 BM25 关键词检索，解决跨语言词汇不匹配的问题。

```
"什么是机器学习？" → 🌐 翻译 → "What is machine learning?" → BM25 检索 → 5 条结果
```

翻译结果会缓存，相同查询不重复调用 API，翻译失败自动降级为原文检索。

### 📊 检索策略对比

在同一界面中对比 **BM25 关键词检索**、**向量语义检索** 和 **混合检索（RRF 融合）** 三种策略的结果，直观感受不同方法的差异。

---

## 演示视频

[📺 观看项目演示视频](rag-lab-demo.mp4)

---

## 扩展方向

- [x] ~~Query 翻译（跨语言 BM25 检索）~~
- [ ] Re-Ranker 重排序（如 BGE-Reranker）
- [ ] 多轮对话记忆
- [ ] HyDE 查询增强
- [ ] LangChain / LlamaIndex 集成对比
- [ ] 流式输出（Streaming）
- [ ] 更多文档格式支持（Markdown、Word）

---

## License

MIT License — 自由使用、修改和分发。详见 [LICENSE](LICENSE)。
