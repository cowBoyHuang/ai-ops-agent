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
_MAX_CODE_TARGETS = 12
_LINE_WINDOW_RADIUS = 18
_APP_CODE_GIT_URL_MAP = {
    "f_tts_trade_order": "http://gitlab.corp.qunar.com/flightdev-tts/tts_trade_order.git",
    "f_tts_trade_core": "http://gitlab.corp.qunar.com/flightdev-tts/tts-trade-core",
}
_CLASS_FILE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\.java\b")
_STACK_CLASS_METHOD_PATTERN = re.compile(r"\b(?:[a-z][a-z0-9_]*\.)+([A-Z][A-Za-z0-9_]+)\.([a-z][A-Za-z0-9_]*)\(")
_SIMPLE_CLASS_LINE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+):\d+\b")
_STACK_FILE_LINE_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+\.(?:java|kt|py)):(\d+)\b")
_FAILURE_PRIORITY_TOKENS = (
    "success\":false",
    "resultok\":false",
    "errorcode",
    "errormsg",
    "refsuberrmsg",
    "校验不通过",
    "失败",
    "block_reason",
    "failres",
)


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
    return (
        "repository not found" in text
        or "clone first" in text
        or "repository invalid" in text
        or "reclone required" in text
    )


def _refresh_repo_before_read(
    *,
    tool_name: str,
    effective_git_url: str,
    repo_name: str,
    mapped_git_url: str,
) -> dict[str, Any]:
    """Pull or clone before reading any local repository content."""
    if tool_name == "code_clone":
        if not effective_git_url:
            return {"ok": False, "message": "missing git_url for clone"}
        return clone_repo(git_url=effective_git_url)

    if effective_git_url:
        refresh_result = pull_repo(git_url=effective_git_url)
        if not bool(refresh_result.get("ok")) and _is_repo_not_found(refresh_result):
            clone_result = clone_repo(git_url=effective_git_url)
            if bool(clone_result.get("ok")):
                return clone_result
        return refresh_result

    if repo_name:
        refresh_result = pull_repo_local(repo_name=repo_name)
        if not bool(refresh_result.get("ok")) and _is_repo_not_found(refresh_result) and mapped_git_url:
            clone_result = clone_repo(git_url=mapped_git_url)
            if bool(clone_result.get("ok")):
                return clone_result
        return refresh_result

    return {"ok": False, "message": "missing git_url/repo_name"}


def _class_name_from_file_name(file_name: str) -> str:
    return Path(str(file_name or "").strip()).stem


def _score_code_hint_row(text: str) -> int:
    lowered = str(text or "").strip().lower()
    score = 0
    for token in _FAILURE_PRIORITY_TOKENS:
        if token in lowered:
            score += 3
    if ".java:" in lowered or ".kt:" in lowered or ".py:" in lowered:
        score += 1
    return score


def _extract_code_hints(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    execution_history = dict(state.get("execution_history") or {})
    code_targets: list[dict[str, Any]] = []
    class_names: list[str] = []
    terms: list[str] = []
    seen_target: set[tuple[str, int]] = set()
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
            row_score = _score_code_hint_row(text)
            for file_name, line_text in _STACK_FILE_LINE_PATTERN.findall(text):
                line_no = int(str(line_text).strip() or "0")
                target_key = (str(file_name).strip(), line_no)
                if target_key not in seen_target:
                    seen_target.add(target_key)
                    code_targets.append(
                        {
                            "file_name": str(file_name).strip(),
                            "class_name": _class_name_from_file_name(file_name),
                            "line_no": line_no,
                            "score": row_score,
                        }
                    )
            for class_name, line_text in re.findall(r"\b([A-Z][A-Za-z0-9_]+):(\d+)\b", text):
                line_no = int(str(line_text).strip() or "0")
                target_key = (f"{str(class_name).strip()}.java", line_no)
                if target_key not in seen_target:
                    seen_target.add(target_key)
                    code_targets.append(
                        {
                            "file_name": f"{str(class_name).strip()}.java",
                            "class_name": str(class_name).strip(),
                            "line_no": line_no,
                            "score": row_score,
                        }
                    )
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

    ranked_targets = sorted(
        code_targets,
        key=lambda item: (-int(item.get("score") or 0), str(item.get("class_name") or ""), int(item.get("line_no") or 0)),
    )
    return ranked_targets[:_MAX_CODE_TARGETS], class_names[:_MAX_CLASS_HINTS], terms[:_MAX_CLASS_HINTS]


def _slice_relevant_content(content: str, terms: list[str], *, line_no: int | None = None) -> str:
    if not content:
        return ""
    if line_no and line_no > 0:
        rows = content.splitlines()
        if rows and line_no <= len(rows):
            start = max(1, line_no - _LINE_WINDOW_RADIUS)
            end = min(len(rows), line_no + _LINE_WINDOW_RADIUS)
            return "\n".join(f"{idx}: {rows[idx - 1]}" for idx in range(start, end + 1))
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


def _find_candidate_paths(root: Path, *, file_name: str = "", class_name: str = "") -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _append(path: Path) -> None:
        if path in seen or not path.is_file():
            return
        seen.add(path)
        candidates.append(path)

    file_name = str(file_name or "").strip()
    class_name = str(class_name or "").strip()
    if file_name:
        for path in root.rglob(file_name):
            _append(path)
    if not candidates and class_name:
        for ext in (".java", ".kt", ".py"):
            for path in root.rglob(f"{class_name}{ext}"):
                _append(path)
    return candidates


def _collect_code_snippets(
    target_dir: str,
    *,
    class_hints: list[str],
    terms: list[str],
    code_targets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(str(target_dir or "")).expanduser()
    if not root.is_dir():
        return []
    normalized_targets = [dict(item or {}) for item in list(code_targets or [])]
    if not normalized_targets and not class_hints:
        return []

    rows: list[dict[str, Any]] = []
    candidate_specs: list[dict[str, Any]] = []
    spec_index_by_path: dict[Path, int] = {}

    def _append_candidate(path: Path, *, line_no: int | None = None) -> None:
        normalized_line = int(line_no or 0) or None
        existing_index = spec_index_by_path.get(path)
        if existing_index is not None:
            if normalized_line and not candidate_specs[existing_index].get("line_no"):
                candidate_specs[existing_index]["line_no"] = normalized_line
            return
        spec_index_by_path[path] = len(candidate_specs)
        candidate_specs.append({"path": path, "line_no": normalized_line})

    for target in normalized_targets:
        file_name = str(target.get("file_name") or "").strip()
        class_name = str(target.get("class_name") or "").strip()
        line_no = int(target.get("line_no") or 0) or None
        for path in _find_candidate_paths(root, file_name=file_name, class_name=class_name):
            _append_candidate(path, line_no=line_no)
            if len(candidate_specs) >= _MAX_FILES:
                break
        if len(candidate_specs) >= _MAX_FILES:
            break

    if len(candidate_specs) < _MAX_FILES:
        for class_name in class_hints:
            cn = str(class_name or "").strip()
            if not cn:
                continue
            for path in _find_candidate_paths(root, class_name=cn):
                _append_candidate(path)
                if len(candidate_specs) >= _MAX_FILES:
                    break
            if len(candidate_specs) >= _MAX_FILES:
                break

    for spec in candidate_specs[:_MAX_FILES]:
        path = Path(spec["path"])
        line_no = spec.get("line_no")
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        snippet = _slice_relevant_content(content, terms, line_no=line_no)[:_MAX_FILE_CHARS]
        if not snippet:
            continue
        rows.append(
            {
                "path": str(path),
                "content": snippet,
                "anchor_line": line_no,
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
    tool_name = str(step.get("tool_name") or "code_pull")
    params = dict(step.get("params") or {})
    git_url = _extract_git_url(params, structured_context)
    repo_name = _extract_repo_name(params, structured_context)
    mapped_git_url = _mapped_git_url(repo_name)
    effective_git_url = git_url or mapped_git_url

    tool_result = _refresh_repo_before_read(
        tool_name=tool_name,
        effective_git_url=effective_git_url,
        repo_name=repo_name,
        mapped_git_url=mapped_git_url,
    )

    if not bool(tool_result.get("ok")):
        return {
            "tool": tool_name,
            "ok": False,
            "error": str(tool_result.get("message") or "code operation failed"),
            "evidence": [],
            "tool_payload": tool_result,
        }

    target_dir = str(tool_result.get("target_dir") or "")
    code_targets, class_hints, hint_terms = _extract_code_hints(state)
    terms = [*class_hints, *hint_terms]
    snippets = _collect_code_snippets(
        target_dir,
        class_hints=class_hints,
        terms=terms,
        code_targets=code_targets,
    )
    extracted = _summarize_code(
        tool_name,
        effective_git_url or repo_name,
        snippets,
        class_hints=class_hints,
        terms=terms,
    )

    evidence = [
        (
            "repo_refresh:"
            f" action={tool_result.get('action') or tool_name}"
            f" status={tool_result.get('status') or 'unknown'}"
            f" target_dir={target_dir}"
        ),
        f"{tool_name} success: {target_dir}",
        f"[summary] {str(extracted.get('summary') or '')}",
    ]
    for item in snippets[:3]:
        anchor_line = int(item.get("anchor_line") or 0)
        if anchor_line > 0:
            evidence.append(f"code_file: {item.get('path')}:{anchor_line}")
        else:
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
