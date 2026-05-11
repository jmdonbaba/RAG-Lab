import gradio as gr

from config import config
from src.pipeline import RAGPipeline

_pipeline_cache = None


def get_pipeline():
    """返回已构建好的 pipeline 单例，避免每次请求重建索引。"""
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = RAGPipeline()
        _pipeline_cache.build_index()
    return _pipeline_cache


def get_fresh_pipeline():
    """创建全新的 pipeline（用于评估等会修改 pipeline 状态的操作）。"""
    pipeline = RAGPipeline()
    pipeline.build_index()
    return pipeline


# ── Theme ──────────────────────────────────────────────────────────
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 3px 0 rgba(0,0,0,0.06)",
    block_radius="12px",
    button_large_radius="8px",
    button_small_radius="8px",
    input_radius="8px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    checkbox_background_color_selected="*primary_600",
    loader_color="*primary_600",
)

# ── CSS ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
footer { display: none !important; }
.sidebar-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stat-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--color-primary-600);
}
.stat-label {
    font-size: 12px;
    color: var(--color-neutral-600);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.source-card {
    background: var(--color-neutral-50);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 4px solid var(--color-primary-400);
}
.source-card .score {
    display: inline-block;
    background: var(--color-primary-100);
    color: var(--color-primary-700);
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}
.method-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    height: 100%;
}
.method-card h3 {
    margin-top: 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--color-neutral-200);
}
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 12px 16px;
    color: #991b1b;
}
"""

# ── Chat helpers ───────────────────────────────────────────────────


def format_sources_html(docs, retrieval_time_ms):
    """Build an HTML block displaying retrieved sources."""
    html = f'<div style="margin-top: 16px;"><div style="font-size: 13px; color: var(--color-neutral-500); margin-bottom: 8px;">'
    html += f"📖 检索了 {len(docs)} 个文档片段 · ⏱ {retrieval_time_ms}ms"
    html += "</div>"
    for doc in docs:
        score_pct = f"{doc['score']:.0%}" if isinstance(doc["score"], float) and doc["score"] <= 1 else str(doc["score"])
        html += (
            f'<div class="source-card">'
            f'<div style="font-weight:600;margin-bottom:4px;">{doc["source"]} '
            f'<span class="score">相关度 {score_pct}</span></div>'
            f'<div style="font-size:13px;color:var(--color-neutral-600);">{doc["content_preview"]}...</div>'
            f"</div>"
        )
    html += "</div>"
    return html


def chat(message: str, history: list):
    """Handle chat — returns generator of (answer, sources_html)."""
    if not message.strip():
        yield "", ""
        return

    try:
        pipeline = get_pipeline()
        result = pipeline.query(message, top_k=config.top_k, use_llm=True)
        answer = result["answer"]
        sources_html = format_sources_html(result["retrieved_docs"], result.get("retrieval_time_ms", 0))
        yield answer, sources_html
    except Exception as e:
        error_html = f'<div class="error-box">❌ 生成回答失败: {e}</div>'
        yield f"抱歉，生成回答时出现错误。请稍后重试。", error_html


# ── Compare ────────────────────────────────────────────────────────

METHOD_META = {
    "bm25": {"name": "BM25 关键词检索", "icon": "🔤", "color": "#7c3aed"},
    "vector": {"name": "向量语义检索", "icon": "🧠", "color": "#2563eb"},
    "hybrid": {"name": "混合检索 (RRF)", "icon": "⚡", "color": "#059669"},
}


def compare_strategies(query: str):
    if not query.strip():
        yield "", "", "", ""
        return

    yield "⏳ 正在检索中...", "", "", ""

    try:
        pipeline = get_pipeline()
        results = pipeline.compare_retrieval_methods(query)

        def build_col(method):
            docs = results.get(method, [])
            meta = METHOD_META[method]
            html = f'<div class="method-card">'
            html += f'<h3 style="color:{meta["color"]}">{meta["icon"]} {meta["name"]}</h3>'
            if method == "bm25" and results.get("translated_query"):
                html += (
                    f'<p style="font-size:12px;color:var(--color-primary-600);margin:4px 0;">'
                    f'🌐 已翻译为: <strong>{results["translated_query"]}</strong></p>'
                )
            html += f'<p style="color:var(--color-neutral-500);font-size:13px;">共 {len(docs)} 条结果</p>'
            for i, d in enumerate(docs, 1):
                score = f"{d['score']:.4f}" if isinstance(d["score"], float) else str(d["score"])
                html += (
                    f'<div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--color-neutral-100);">'
                    f'<div style="font-size:13px;font-weight:600;">{i}. {d["source"]}</div>'
                    f'<div style="font-size:12px;color:var(--color-neutral-500);">相关度: {score}</div>'
                    f'<div style="font-size:12px;color:var(--color-neutral-600);">{d.get("preview", "")}...</div>'
                    f"</div>"
                )
            html += "</div>"
            return html

        yield "", build_col("bm25"), build_col("vector"), build_col("hybrid")
    except Exception as e:
        err = f'<div class="error-box">❌ 检索对比失败: {e}</div>'
        yield "", err, err, err


# ── Evaluate ───────────────────────────────────────────────────────


def evaluate_system(progress=gr.Progress()):
    progress(0.05, desc="正在初始化流水线...")
    pipeline = get_fresh_pipeline()

    progress(0.15, desc="正在运行评估（可能需要 1-2 分钟）...")
    from src.evaluator import Evaluator

    evaluator = Evaluator(pipeline)
    results = evaluator.run_full_evaluation()

    # Build summary cards
    summary = {}
    if "retrieval" in results:
        r = results["retrieval"]
        summary["检索查询数"] = str(r.get("num_queries", "N/A"))
        summary["平均检索延迟"] = f'{r.get("avg_latency_ms", "N/A")} ms'
        summary["BM25-向量重叠度"] = str(r.get("avg_bm25_vector_overlap", "N/A"))
    if "generation" in results:
        g = results["generation"]
        summary["平均回答长度"] = f'{g.get("avg_answer_length", 0):.0f} 字符'
        summary["平均来源数"] = f'{g.get("avg_sources_used", 0):.1f}'

    progress(0.5, desc="处理分块策略对比...")

    # Chunking comparison table — 用 list-of-lists 格式，确保值都是 Python 原生类型
    chunking_data = []
    if "chunking_comparison" in results:
        for strat, d in results["chunking_comparison"].items():
            chunking_data.append([
                strat,
                int(d.get("num_chunks", 0)),
                f'{float(d.get("avg_chunk_length", 0)):.0f}',
                f'{float(d.get("avg_latency_ms", 0)):.0f} ms',
                f'{float(d.get("avg_bm25_vector_overlap", 0)):.4f}',
            ])

    progress(0.7, desc="处理嵌入模型对比...")

    # Embedding comparison table — 同上
    embedding_data = []
    if "embedding_comparison" in results:
        for model, d in results["embedding_comparison"].items():
            embedding_data.append([
                model,
                int(d.get("dim", 0)),
                f'{float(d.get("avg_similarity", 0)):.4f}',
                f'{float(d.get("coverage", 0)):.2%}',
            ])

    progress(0.95, desc="完成!")

    return summary, chunking_data, embedding_data


# ── Sidebar info ───────────────────────────────────────────────────


def get_system_info():
    return f"""
| 配置项 | 值 |
|--------|-----|
| 嵌入模型 | `{config.default_embedding_model}` |
| 分块大小 | {config.chunk_size} |
| 分块重叠 | {config.chunk_overlap} |
| Top-K | {config.top_k} |
| 向量存储 | {config.vector_store_type} |
| LLM 提供商 | {config.llm_provider} |
| LLM 模型 | {config.deepseek_model if config.llm_provider == 'deepseek' else config.openai_model} |
| 融合算法 | {config.hybrid_fusion} |
"""


# ── Main UI ────────────────────────────────────────────────────────


def build_ui():
    with gr.Blocks(
        title="RAG 知识问答系统",
    ) as demo:
        # ── Header ────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:24px 0 8px 0;">
            <h1 style="font-size:28px;font-weight:800;margin:0;background:linear-gradient(135deg, #2563eb, #7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                RAG 知识问答系统
            </h1>
            <p style="color:#6b7280;font-size:14px;margin-top:6px;">
                基于 CMU 10-701 机器学习课程讲义 · 检索增强生成动手实验
            </p>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Sidebar ───────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.HTML("""
                <div class="sidebar-card">
                    <div style="font-weight:700;font-size:16px;margin-bottom:12px;">系统配置</div>
                """)
                sys_info = gr.Markdown(get_system_info(), elem_classes=["sidebar-card"])
                gr.HTML("</div>")

                gr.HTML("""
                <div class="sidebar-card">
                    <div style="font-weight:700;font-size:16px;margin-bottom:8px;">关于 RAG</div>
                    <p style="font-size:13px;color:#6b7280;line-height:1.6;">
                    RAG（检索增强生成）通过<strong>先检索、再生成</strong>的方式，
                    让 LLM 基于真实文档回答问题，有效减少幻觉。
                    </p>
                    <div style="display:flex;gap:8px;margin-top:12px;font-size:12px;">
                        <span style="background:#e0e7ff;color:#3730a3;padding:4px 10px;border-radius:99px;">ChromaDB</span>
                        <span style="background:#fce7f3;color:#9d174d;padding:4px 10px;border-radius:99px;">BM25</span>
                        <span style="background:#d1fae5;color:#065f46;padding:4px 10px;border-radius:99px;">FAISS</span>
                    </div>
                </div>
                """)

            # ── Main Content ──────────────────────────────────
            with gr.Column(scale=3, min_width=480):
                with gr.Tabs():
                    # ── Tab 1: Chat ───────────────────────────
                    with gr.TabItem("💬 智能问答"):
                        with gr.Row():
                            with gr.Column(scale=4):
                                chatbot = gr.ChatInterface(
                                    fn=chat,
                                    additional_outputs=[
                                        gr.HTML(label="检索来源", elem_classes=["source-area"]),
                                    ],
                                    title="",
                                    description="向 AI 助教提问机器学习相关问题，回答将附带 CMU 讲义来源。",
                                    examples=[
                                        "什么是逻辑回归？",
                                        "请解释偏差-方差权衡",
                                        "梯度下降是如何工作的？",
                                        "什么是EM算法？",
                                        "请解释支持向量机的原理",
                                        "贝叶斯推断是什么？",
                                    ],
                                    autofocus=True,
                                )

                    # ── Tab 2: Compare ────────────────────────
                    with gr.TabItem("🔍 检索对比"):
                        gr.Markdown("""
                        ### 对比三种检索策略
                        同一个查询分别执行 BM25 关键词检索、向量语义检索 和混合检索（RRF 融合），直观感受不同策略的差异。
                        """)
                        with gr.Row():
                            compare_query = gr.Textbox(
                                label="查询内容",
                                placeholder="输入查询内容，对比不同检索方法的效果...",
                                scale=3,
                            )
                            compare_btn = gr.Button("开始对比", variant="primary", scale=1)
                        compare_status = gr.Markdown("", elem_id="compare-loading")

                        with gr.Row(equal_height=True):
                            bm25_col = gr.HTML(label="BM25 关键词检索")
                            vec_col = gr.HTML(label="向量语义检索")
                            hyb_col = gr.HTML(label="混合检索 (RRF)")

                        compare_btn.click(
                            fn=compare_strategies,
                            inputs=[compare_query],
                            outputs=[compare_status, bm25_col, vec_col, hyb_col],
                        )

                    # ── Tab 3: Evaluate ───────────────────────
                    with gr.TabItem("📊 系统评估"):
                        gr.Markdown("""
                        ### 全面系统评估
                        运行覆盖分块策略、嵌入模型、检索性能、生成质量的完整评估。首次运行可能需要 1-2 分钟。
                        """)

                        eval_btn = gr.Button("▶ 运行完整评估", variant="primary", size="lg")
                        eval_status = gr.Markdown("")

                        with gr.Group(visible=False) as eval_results_group:
                            gr.Markdown("#### 📋 概览")
                            eval_summary = gr.JSON(label="评估概览")

                            gr.Markdown("#### 📐 分块策略对比")
                            chunking_table = gr.Dataframe(
                                headers=["策略", "分块数", "平均长度", "延迟 (ms)", "重叠度"],
                                label="分块策略对比",
                                interactive=False,
                                wrap=True,
                            )

                            gr.Markdown("#### 🧬 嵌入模型对比")
                            embedding_table = gr.Dataframe(
                                headers=["模型", "维度", "平均相似度", "覆盖率"],
                                label="嵌入模型对比",
                                interactive=False,
                                wrap=True,
                            )

                        def on_eval_click(progress=gr.Progress()):
                            # First yield: immediate feedback so the button doesn't look dead
                            yield (
                                gr.update(visible=False),
                                None, None, None,
                                "⏳ 正在初始化评估流水线...\n\n> 首次运行需要 1-2 分钟，请耐心等待。",
                            )
                            try:
                                summary, chunking, embedding = evaluate_system(progress)
                                yield (
                                    gr.update(visible=True),
                                    summary if summary else None,
                                    chunking,
                                    embedding,
                                    "✅ 评估完成！",
                                )
                            except Exception as exc:
                                import traceback
                                yield (
                                    gr.update(visible=False),
                                    None, None, None,
                                    f"❌ 评估失败:\n```\n{traceback.format_exc()}\n```",
                                )

                        eval_btn.click(
                            fn=on_eval_click,
                            inputs=[],
                            outputs=[eval_results_group, eval_summary, chunking_table, embedding_table, eval_status],
                        )

                    # ── Tab 4: About ──────────────────────────
                    with gr.TabItem("ℹ️ 关于系统"):
                        gr.Markdown(f"""
                        ## 关于本系统

                        本 RAG（检索增强生成）系统完整展示了从文档抓取到答案生成的完整流水线。

                        ### 技术栈
                        | 层级 | 技术 |
                        |------|------|
                        | 文档解析 | pypdf + BeautifulSoup |
                        | 文本分块 | 固定Token / 递归字符 / 语义分块 |
                        | 嵌入模型 | Sentence-Transformers（当前: `{config.default_embedding_model}`） |
                        | 向量存储 | ChromaDB / FAISS |
                        | 检索方式 | BM25 + 向量相似度 → RRF 混合检索 |
                        | 答案生成 | {config.llm_provider.upper()} API（{config.deepseek_model if config.llm_provider == 'deepseek' else config.openai_model}） |

        """)

    return demo


def main():
    print("正在初始化 RAG 流水线...")
    get_pipeline()

    print("正在启动 Gradio 网页界面...")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=custom_theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
