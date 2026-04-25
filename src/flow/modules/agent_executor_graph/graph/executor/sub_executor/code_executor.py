"""代码子执行器：拉取代码并让大模型读取关键信息。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.llm import chat_with_llm
from tool.code_tool import clone_repo, pull_repo

_MAX_FILES = 6
_MAX_FILE_CHARS = 2000


def _extract_git_url(params: dict[str, Any], structured_context: dict[str, Any]) -> str:
    value = (
        params.get("git_url")
        or params.get("repo_url")
        or structured_context.get("git_url")
        or dict(structured_context.get("code_repo") or {}).get("git_url")
        or ""
    )
    return str(value).strip()


def _collect_code_snippets(target_dir: str) -> list[dict[str, str]]:
    root = Path(str(target_dir or "")).expanduser()
    if not root.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py"))[:_MAX_FILES]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "content": content[:_MAX_FILE_CHARS],
            }
        )
    return rows


def _summarize_code(tool_name: str, git_url: str, snippets: list[dict[str, str]]) -> dict[str, Any]:
    if not snippets:
        return {"summary": "未读取到代码文件", "keywords": [], "facts": {}}
    system_prompt = (
        "你是代码排障助手。请读取代码片段并提取与排障相关关键信息。"
        "返回 JSON，字段：summary(字符串), keywords(字符串数组), facts(对象)。"
    )
    user_prompt = (
        f"工具: {tool_name}\n"
        f"仓库: {git_url}\n"
        f"代码片段:\n{json.dumps(snippets, ensure_ascii=False)}"
    )
    raw = chat_with_llm(question=user_prompt, system_prompt=system_prompt)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary") or "代码阅读完成"),
                "keywords": [str(item).strip() for item in list(parsed.get("keywords") or []) if str(item).strip()],
                "facts": dict(parsed.get("facts") or {}),
            }
    except Exception:
        pass
    return {"summary": "代码阅读完成", "keywords": [], "facts": {}}


def run(*, step: dict[str, Any], state: dict[str, Any], structured_context: dict[str, Any]) -> dict[str, Any]:
    _ = state
    tool_name = str(step.get("tool_name") or "code_pull")
    params = dict(step.get("params") or {})
    git_url = _extract_git_url(params, structured_context)
    if not git_url:
        return {"tool": tool_name, "ok": False, "error": "missing git_url", "evidence": []}

    if tool_name == "code_clone":
        tool_result = clone_repo(git_url=git_url)
    else:
        tool_result = pull_repo(git_url=git_url)

    if not bool(tool_result.get("ok")):
        return {
            "tool": tool_name,
            "ok": False,
            "error": str(tool_result.get("message") or "code operation failed"),
            "evidence": [],
            "tool_payload": tool_result,
        }

    target_dir = str(tool_result.get("target_dir") or "")
    snippets = _collect_code_snippets(target_dir)
    extracted = _summarize_code(tool_name, git_url, snippets)

    evidence = [f"{tool_name} success: {target_dir}", f"[summary] {str(extracted.get('summary') or '')}"]
    for item in snippets[:3]:
        evidence.append(f"code_file: {item.get('path')}")
    return {
        "tool": tool_name,
        "ok": True,
        "error": "",
        "evidence": evidence,
        "effective_info": extracted,
        "code_snippet_count": len(snippets),
        "tool_payload": tool_result,
    }

