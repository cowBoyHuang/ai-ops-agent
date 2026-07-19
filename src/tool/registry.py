"""Single registry for annotated agent tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from tool.code_index_tools import CODE_INDEX_TOOLS
from tool.log_tools import LOG_TOOLS
from tool.rag_tools import RAG_TOOLS

_TOOLS: list[BaseTool] = [*LOG_TOOLS, *CODE_INDEX_TOOLS, *RAG_TOOLS]
_TOOL_BY_NAME: dict[str, BaseTool] = {item.name: item for item in _TOOLS}

if len(_TOOLS) != len(_TOOL_BY_NAME):
    names = [item.name for item in _TOOLS]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    raise RuntimeError(f"duplicate tool names: {duplicated}")


def get_all_tools() -> list[BaseTool]:
    """Return all registered tools."""
    return list(_TOOLS)


def get_tool(name: str) -> BaseTool:
    """Return a registered tool by name."""
    key = str(name or "").strip()
    if key not in _TOOL_BY_NAME:
        raise KeyError(f"unsupported tool: {key}")
    return _TOOL_BY_NAME[key]


def build_tool_schemas_for_prompt() -> list[dict[str, Any]]:
    """Build compact tool schemas for executor prompts."""
    rows: list[dict[str, Any]] = []
    for item in _TOOLS:
        schema = item.args_schema.model_json_schema() if item.args_schema is not None else {}
        rows.append(
            {
                "tool_name": item.name,
                "description": str(item.description or "").strip(),
                "params_schema": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        )
    return rows


def invoke_tool(name: str, args: dict[str, Any]) -> Any:
    """Invoke a registered tool and normalize lookup/execution errors."""
    key = str(name or "").strip()
    try:
        selected = get_tool(key)
    except KeyError:
        return {
            "tool": key,
            "ok": False,
            "error": f"unsupported tool: {key}",
            "evidence": [],
        }
    try:
        return selected.invoke(dict(args or {}))
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": key,
            "ok": False,
            "error": str(exc),
            "evidence": [],
        }
