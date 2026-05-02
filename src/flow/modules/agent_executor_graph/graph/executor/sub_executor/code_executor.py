"""代码子执行器：拉取代码并让大模型读取关键信息。"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from llm.llm import chat_with_llm
from tool.code_tool import clone_repo, pull_repo, pull_repo_local

_MAX_FILES = 6
_MAX_FILE_CHARS = 2000
_MAX_CLASS_HINTS = 20
_SUPPORTED_GLOBS = ("*.py", "*.java", "*.kt", "*.xml", "*.yml", "*.yaml", "*.properties")
_APP_CODE_GIT_URL_MAP = {
    "f_tts_trade_order": "http://gitlab.corp.qunar.com/flightdev-tts/tts_trade_order.git",
}
_CLASS_FILE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\.java\b")
_STACK_CLASS_METHOD_PATTERN = re.compile(r"\b(?:[a-z][a-z0-9_]*\.)+([A-Z][A-Za-z0-9_]+)\.([a-z][A-Za-z0-9_]*)\(")
_SIMPLE_CLASS_LINE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+):\d+\b")


def _is_placeholder_git_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    placeholder_tokens = ("待补充", "todo", "to be filled", "tbd", "placeholder")
    return any(token in text for token in placeholder_tokens)


def _extract_git_url(params: dict[str, Any], structured_context: dict[str, Any]) -> str:
    value = (
        params.get("git_url")
        or params.get("repo_url")
        or structured_context.get("git_url")
        or dict(structured_context.get("code_repo") or {}).get("git_url")
        or ""
    )
    text = str(value).strip()
    if _is_placeholder_git_url(text):
        return ""
    return text


def _extract_repo_name(params: dict[str, Any], structured_context: dict[str, Any]) -> str:
    value = (
        params.get("repo_name")
        or params.get("app_code")
        or structured_context.get("app_code")
        or dict(structured_context.get("code_repo") or {}).get("repo_name")
        or ""
    )
    return str(value).strip()


def _mapped_git_url(repo_name: str) -> str:
    return str(_APP_CODE_GIT_URL_MAP.get(str(repo_name or "").strip().lower()) or "").strip()


def _is_repo_not_found(result: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(result.get("message") or ""),
            str(result.get("stderr") or ""),
            str(result.get("status") or ""),
        ]
    ).lower()
    return "repository not found" in text or "clone first" in text


def _extract_code_hints(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    execution_history = dict(state.get("execution_history") or {})
    class_names: list[str] = []
    terms: list[str] = []
    seen_class: set[str] = set()
    seen_terms: set[str] = set()

    keys = sorted(execution_history.keys())
    for key in keys:
        item = dict(execution_history.get(key) or {})
        raw_result = dict(item.get("raw_result") or {})
        for row in list(raw_result.get("evidence") or []):
            text = str(row or "")
            if not text:
                continue
            for matched in _CLASS_FILE_PATTERN.findall(text):
                name = str(matched).strip()
                if name and name not in seen_class:
                    seen_class.add(name)
                    class_names.append(name)
            for class_name, method_name in _STACK_CLASS_METHOD_PATTERN.findall(text):
                cn = str(class_name).strip()
                mn = str(method_name).strip()
                if cn and cn not in seen_class:
                    seen_class.add(cn)
                    class_names.append(cn)
                if mn and mn not in seen_terms:
                    seen_terms.add(mn)
                    terms.append(mn)
            for matched in _SIMPLE_CLASS_LINE_PATTERN.findall(text):
                name = str(matched).strip()
                if name and name not in seen_class:
                    seen_class.add(name)
                    class_names.append(name)

    return class_names[:_MAX_CLASS_HINTS], terms[:_MAX_CLASS_HINTS]


def _slice_relevant_content(content: str, terms: list[str]) -> str:
    if not content:
        return ""
    haystack = content.lower()
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        idx = haystack.find(text.lower())
        if idx < 0:
            continue
        left = max(0, idx - 600)
        right = min(len(content), idx + 1200)
        return content[left:right]
    return content[:_MAX_FILE_CHARS]


def _collect_code_snippets(target_dir: str, *, class_hints: list[str], terms: list[str]) -> list[dict[str, str]]:
    root = Path(str(target_dir or "")).expanduser()
    if not root.is_dir():
        return []
    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    candidate_paths: list[Path] = []

    def _append_candidate(path: Path) -> None:
        if path in seen_paths:
            return
        seen_paths.add(path)
        candidate_paths.append(path)

    for class_name in class_hints:
        cn = str(class_name or "").strip()
        if not cn:
            continue
        for ext in (".java", ".kt", ".py"):
            pattern = f"*{cn}{ext}"
            for path in root.rglob(pattern):
                if path.is_file():
                    _append_candidate(path)
                if len(candidate_paths) >= _MAX_FILES:
                    break
            if len(candidate_paths) >= _MAX_FILES:
                break
        if len(candidate_paths) >= _MAX_FILES:
            break

    if len(candidate_paths) < _MAX_FILES:
        for glob_pattern in _SUPPORTED_GLOBS:
            for path in root.rglob(glob_pattern):
                if path.is_file():
                    _append_candidate(path)
                if len(candidate_paths) >= _MAX_FILES:
                    break
            if len(candidate_paths) >= _MAX_FILES:
                break

    for path in candidate_paths[:_MAX_FILES]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "content": _slice_relevant_content(content, terms)[:_MAX_FILE_CHARS],
            }
        )
    return rows


def _summarize_code(
    tool_name: str,
    git_url: str,
    snippets: list[dict[str, str]],
    *,
    class_hints: list[str],
    terms: list[str],
) -> dict[str, Any]:
    if not snippets:
        return {"summary": "未读取到代码文件", "keywords": [], "facts": {}}
    system_prompt = (
        "你是代码排障助手。请读取代码片段并提取与排障相关关键信息。"
        "返回 JSON，字段：summary(字符串), keywords(字符串数组), facts(对象)。"
    )
    user_prompt = (
        f"工具: {tool_name}\n"
        f"仓库: {git_url}\n"
        f"日志类名线索: {json.dumps(class_hints, ensure_ascii=False)}\n"
        f"日志关键词线索: {json.dumps(terms, ensure_ascii=False)}\n"
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
    repo_name = _extract_repo_name(params, structured_context)
    mapped_git_url = _mapped_git_url(repo_name)
    effective_git_url = git_url or mapped_git_url

    if tool_name == "code_clone":
        if not effective_git_url:
            return {"tool": tool_name, "ok": False, "error": "missing git_url for clone", "evidence": []}
        tool_result = clone_repo(git_url=effective_git_url)
    else:
        if effective_git_url:
            tool_result = pull_repo(git_url=effective_git_url)
            if not bool(tool_result.get("ok")) and _is_repo_not_found(tool_result):
                clone_result = clone_repo(git_url=effective_git_url)
                if bool(clone_result.get("ok")):
                    tool_result = clone_result
        elif repo_name:
            tool_result = pull_repo_local(repo_name=repo_name)
            if not bool(tool_result.get("ok")) and _is_repo_not_found(tool_result) and mapped_git_url:
                clone_result = clone_repo(git_url=mapped_git_url)
                if bool(clone_result.get("ok")):
                    tool_result = clone_result
        else:
            return {"tool": tool_name, "ok": False, "error": "missing git_url/repo_name", "evidence": []}

    if not bool(tool_result.get("ok")):
        return {
            "tool": tool_name,
            "ok": False,
            "error": str(tool_result.get("message") or "code operation failed"),
            "evidence": [],
            "tool_payload": tool_result,
        }

    target_dir = str(tool_result.get("target_dir") or "")
    class_hints, hint_terms = _extract_code_hints(state)
    terms = [*class_hints, *hint_terms]
    snippets = _collect_code_snippets(target_dir, class_hints=class_hints, terms=terms)
    extracted = _summarize_code(
        tool_name,
        effective_git_url or repo_name,
        snippets,
        class_hints=class_hints,
        terms=terms,
    )

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
