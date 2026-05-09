"""
RAG-Lab 命令行入口

提供 5 个子命令，对应 RAG 学习的不同阶段:
  build    → 构建索引（第1-2课）
  query    → 单次问答（第3课）
  compare  → 对比检索策略（第3课）
  evaluate → 运行评估（第4课）
  ui       → 启动网页界面

快速上手:
  python main.py build              # 第一步：构建索引
  python main.py query "什么是SVM?"  # 第二步：提问
  python main.py ui                 # 第三步：打开网页体验
"""

import argparse
from config import config


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Lab — 检索增强生成动手实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py build                                    # 默认参数构建索引
  python main.py build --strategy semantic --store faiss  # 语义分块 + FAISS
  python main.py query "什么是逻辑回归？"                   # 问答
  python main.py query "SVM原理" --top_k 3 --no_llm       # 仅检索
  python main.py compare "贝叶斯推断"                      # 对比检索方法
  python main.py evaluate                                 # 运行评估
  python main.py ui                                       # 启动网页界面
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ---- build: 构建索引 ----
    build_parser = subparsers.add_parser("build", help="构建搜索索引（文档→分块→嵌入→存储）")
    build_parser.add_argument(
        "--strategy", default="recursive_char",
        choices=["fixed_token", "recursive_char", "semantic"],
        help="分块策略: fixed_token(按Token), recursive_char(递归字符,默认), semantic(按句子)"
    )
    build_parser.add_argument(
        "--store", default="chroma", choices=["chroma", "faiss"],
        help="向量存储: chroma(持久化,默认), faiss(纯内存,更快)"
    )
    build_parser.add_argument(
        "--embedding", default=config.default_embedding_model,
        help=f"嵌入模型 (默认: {config.default_embedding_model})"
    )

    # ---- query: 单次查询 ----
    query_parser = subparsers.add_parser("query", help="执行单次 RAG 查询")
    query_parser.add_argument("question", help="要提问的问题")
    query_parser.add_argument("--top_k", type=int, default=5, help="返回文档数 (默认5)")
    query_parser.add_argument("--no_llm", action="store_true",
                              help="仅检索不生成（调试用）")

    # ---- evaluate: 运行评估 ----
    eval_parser = subparsers.add_parser("evaluate", help="运行完整系统评估")

    # ---- compare: 对比检索方法 ----
    compare_parser = subparsers.add_parser("compare", help="对比 BM25 / 向量 / 混合检索效果")
    compare_parser.add_argument("question", help="要提问的问题")

    # ---- ui: Web 界面 ----
    ui_parser = subparsers.add_parser("ui", help="启动 Gradio 网页交互界面")

    args = parser.parse_args()

    # ---- 执行命令 ----
    if args.command == "build":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(
            strategy=args.strategy,
            store_type=args.store,
            embedding_model=args.embedding,
        )
        pipeline.build_index()
        print(f"\n索引构建成功！策略={args.strategy}, 存储={args.store}")

    elif args.command == "query":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.build_index()
        result = pipeline.query(args.question, top_k=args.top_k, use_llm=not args.no_llm)

        print(f"\n{'='*60}")
        print(f"问题: {result['query']}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        print(f"\n--- 参考来源 ({len(result['sources'])}) ---")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s}")
        print(f"\n检索耗时: {result['retrieval_time_ms']}ms")

    elif args.command == "evaluate":
        from src.pipeline import RAGPipeline
        from src.evaluator import Evaluator
        pipeline = RAGPipeline()
        pipeline.build_index()
        evaluator = Evaluator(pipeline)
        evaluator.run_full_evaluation()
        evaluator.print_report()

    elif args.command == "compare":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline.build_index()
        results = pipeline.compare_retrieval_methods(args.question)
        method_names = {"hybrid": "混合检索", "bm25": "BM25关键词检索", "vector": "向量语义检索"}
        for method in ["hybrid", "bm25", "vector"]:
            print(f"\n--- {method_names[method]} ---")
            for i, d in enumerate(results[method], 1):
                print(f"  [{i}] {d['source']} (相关度: {d['score']})")

    elif args.command == "ui":
        from src.app import main as ui_main
        ui_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
