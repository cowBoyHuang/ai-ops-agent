"""脚本说明：rag_ingest/import_local_docs.py。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flow.modules.core.rag.llamaindex_retriever import LlamaIndexRetriever
from config import get_settings


# 方法注释（业务）:
# - 业务：在业务处理链路中负责“normalize、suffixes”。
# - 入参：`raw`(str)=raw参数
# - 出参：`list[str]`，返回当前方法处理后的结果。
# - 逻辑：调用 `lower`、`split`、`strip` 推进主流程；最终返回处理结果。
def _normalize_suffixes(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return [item if item.startswith(".") else f".{item}" for item in values]


# 方法注释（业务）:
# - 业务：在业务处理链路中负责“执行、import”。
# - 入参：`source_dir`(str | Path | None)=sourcedir参数；`suffixes`(list[str] | None)=suffixes参数
# - 出参：`dict`，返回当前方法处理后的结果。
# - 逻辑：调用 `get_settings`、`LlamaIndexRetriever`、`import_local_documents` 推进主流程；最终返回处理结果。
def run_import(*, source_dir: str | Path | None = None, suffixes: list[str] | None = None) -> dict:
    settings = get_settings()
    target_dir = Path(source_dir) if source_dir else settings.knowledge_dir
    # 读取本地文档并导入到本地持久化 RAG 索引目录。
    retriever = LlamaIndexRetriever(
        knowledge_dir=target_dir,
        persist_dir=settings.rag_persist_dir,
    )
    result = retriever.import_local_documents(
        source_dir=target_dir,
        suffixes=suffixes or [".md", ".txt", ".doc", ".docx", ".log", ".rst"],
    )
    return result


# 方法注释（业务）:
# - 业务：在业务处理链路中负责“main”。
# - 入参：无。
# - 出参：`None`，无返回值，主要执行状态更新或副作用。
# - 逻辑：调用 `ArgumentParser`、`add_argument`、`parse_args` 推进主流程；完成状态更新后结束。
def main() -> None:
    # 核心说明：调用 `ArgumentParser`、`add_argument`、`parse_args` 推进主流程；完成状态更新后结束。
    parser = argparse.ArgumentParser(description="导入本地文档到 RAG 索引（LlamaIndex）")
    parser.add_argument(
        "--source-dir",
        default="",
        help="文档目录，默认使用配置 knowledge_dir（rag_ingest/source_docs）",
    )
    parser.add_argument(
        "--suffixes",
        default=".md,.txt,.doc,.docx,.log,.rst",
        help="允许导入的后缀，逗号分隔",
    )
    args = parser.parse_args()
    suffixes = _normalize_suffixes(args.suffixes)
    result = run_import(source_dir=args.source_dir or None, suffixes=suffixes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # 这里做条件分流，便于分别处理不同输入情况。
    main()

