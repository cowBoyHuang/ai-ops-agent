"""Annotated Code Index tools used by the agent runtime."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from tool import code_index_client


@tool("indexProject", description="Code Index 项目索引工具。根据 project_path 请求本地 Code Index Service 建立项目索引。")
def index_project_tool(project_path: str) -> dict[str, Any]:
    """Index a project through the local Code Index service."""
    return code_index_client.index_project(project_path)


@tool("searchMethod", description="Code Index 方法搜索工具。根据关键词检索真实代码方法，返回类名、方法名、签名和文件行号。")
def search_method_tool(keyword: str) -> dict[str, Any]:
    """Search methods by keyword."""
    return code_index_client.search_method(keyword)


@tool("locateCode", description="Code Index 代码定位工具。根据 class_name 与 line 定位当前方法、调用者、被调用者和相关日志。")
def locate_code_tool(class_name: str, line: int) -> dict[str, Any]:
    """Locate code by class and line."""
    return code_index_client.locate_code(class_name, int(line or 0))


@tool(
    "analyzeCodeFromLogs",
    description=(
        "日志证据不足时的代码分析兜底工具。优先使用日志中的类名+行号定位方法，"
        "定位失败再按关键词搜索方法，最后回退本地源码索引；不得伪造代码结论。"
    ),
)
def analyze_code_from_logs_tool(
    question: str,
    evidence_rows: list[str],
    extra_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze code using log evidence and optional keywords."""
    return code_index_client.analyze_code_from_logs(
        question=question,
        evidence_rows=evidence_rows,
        extra_keywords=extra_keywords,
    )


@tool(
    "analyzeCodeForBusinessConsult",
    description=(
        "业务咨询专用代码分析工具。用于在回答业务问题前补充实际代码分析证据，"
        "必须与业务文档证据共同使用。文档与代码冲突时，以当前实际代码为准，并明确标注文档待确认或待更新。"
    ),
)
def analyze_code_for_business_consult_tool(
    question: str,
    structured_context: dict[str, Any] | None = None,
    evidence_rows: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze code for a business consultation answer."""
    return code_index_client.analyze_code_for_business_consult(
        question=question,
        structured_context=structured_context,
        evidence_rows=evidence_rows,
    )


CODE_INDEX_TOOLS = [
    index_project_tool,
    search_method_tool,
    locate_code_tool,
    analyze_code_from_logs_tool,
    analyze_code_for_business_consult_tool,
]
