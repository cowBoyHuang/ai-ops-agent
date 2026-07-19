"""Lightweight local source-code lookup fallback.

This module intentionally avoids external parser dependencies. It provides
best-effort Java symbol and method lookup for cases where the Java code-index
service is unavailable or too method-centric for class/enum declaration
questions.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

_PACKAGE_PATTERN = re.compile(r"^\s*package\s+([A-Za-z_][\w.]*);", re.MULTILINE)
_SYMBOL_PATTERN = re.compile(
    r"^[ \t]*(?:public|protected|private)?[ \t]*(?:abstract[ \t]+|final[ \t]+)?"
    r"(?P<kind>class|interface|enum)\s+(?P<name>[A-Z][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
_METHOD_PATTERN = re.compile(
    r"^[ \t]*(?:@\w+(?:\([^)]*\))?[ \t]*)*"
    r"(?:(?:public|protected|private|static|final|synchronized|abstract|native)\s+)*"
    r"(?P<return>[A-Za-z_][\w<>\[\], ? extends super.]*?)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:throws\s+[^{]+)?\{",
    re.MULTILINE,
)
_TOKEN_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]{2,80})")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,80}")
_QFLOW_XML_COMPONENT_ID_PATTERN = re.compile(r"\bid\s*=\s*[\"']([^\"']+)[\"']")
_QFLOW_COMPONENT_ANNOTATION_PATTERN = re.compile(
    r"@QFlowComponent\s*\(\s*(?:value\s*=\s*)?[\"'](?P<component_id>[^\"']+)[\"']"
)

_BUSINESS_ENTRY_TERMS = (
    "生单",
    "落单",
    "下单",
    "创建订单",
    "订单创建",
    "订单入口",
    "create order",
    "createorder",
)
_CJK_QUERY_STOP_PHRASES = (
    "是哪个文件呢",
    "是哪个文件",
    "哪个文件",
    "哪个入口",
    "调用位置",
    "调用哪个接口",
    "调用哪个",
    "哪个接口",
    "入口",
    "文件",
    "接口",
    "调用",
    "位置",
    "在哪里",
    "在哪",
    "帮我",
    "找到",
    "代码",
    "方法",
    "哪个",
    "什么",
    "是",
    "呢",
    "吗",
)
_BUSINESS_TERM_EXPANSIONS = {
    "生单": ("createOrder", "CreateOrder", "flow-single", "flow-double", "SingleCreateOrderServiceImpl"),
    "落单": ("createOrder", "CreateOrder", "flow-single", "flow-double", "SingleCreateOrderServiceImpl"),
    "下单": ("createOrder", "CreateOrder", "flow-single", "flow-double", "SingleCreateOrderServiceImpl"),
    "创建订单": ("createOrder", "CreateOrder", "flow-single", "flow-double", "SingleCreateOrderServiceImpl"),
    "订单创建": ("createOrder", "CreateOrder", "flow-single", "flow-double", "SingleCreateOrderServiceImpl"),
    "特殊产品": (
        "specialProductRuleComp",
        "specialProductBeforeOrderComp",
        "SpecialProductRuleInterceptor",
        "SpecialProductBeforeOrderInterceptor",
        "BookingNewValidateService",
        "SpecialProductUtil",
    ),
    "特殊产品校验": (
        "specialProductRuleComp",
        "specialProductBeforeOrderComp",
        "SpecialProductRuleInterceptor",
        "SpecialProductBeforeOrderInterceptor",
        "BookingNewValidateService",
        "SpecialProductUtil",
    ),
    "生编": (
        "策略生编",
        "strategyNormalPnrOrderServiceComp",
        "roundPolicyPnrOrderServiceComp",
        "NormalPolicyPnrOrderServiceImpl",
        "RoundPolicyPnrOrderServiceImpl",
        "PnrCreateStrategyFactory",
        "createPnr",
        "pnr",
    ),
    "策略生编": (
        "strategyNormalPnrOrderServiceComp",
        "roundPolicyPnrOrderServiceComp",
        "NormalPolicyPnrOrderServiceImpl",
        "RoundPolicyPnrOrderServiceImpl",
        "PnrCreateStrategyFactory",
        "createPnr",
        "pnr",
    ),
    "支付前校验": ("beforeOrderValidate", "specialProductBeforeOrderComp", "IOrderValidateServiceImpl"),
    "生单前校验": ("beforeOrderValidate", "specialProductBeforeOrderComp", "IOrderValidateServiceImpl"),
    "用户画像": ("userPortrait", "FlightDataUserPortraitChecker", "BookingNewValidateService"),
    "兜底": ("fallback", "back", "RePolicy", "ChangeSupplier", "FailBack"),
}
_ENTRY_METHOD_NAMES = ("execute", "createOrder", "validate", "check", "process", "handle", "apply")
_SOURCE_SUFFIXES = {
    ".java",
    ".kt",
    ".xml",
    ".yml",
    ".yaml",
    ".properties",
    ".conf",
    ".json",
    ".txt",
    ".md",
    ".sql",
}
_MAX_TEXT_MATCHES = 8


def _default_code_root() -> Path:
    return Path(__file__).resolve().parents[1] / "code_repo"


def _split_roots(raw: str) -> list[str]:
    normalized = str(raw or "").replace(",", os.pathsep)
    return [item.strip() for item in normalized.split(os.pathsep) if item.strip()]


def code_roots() -> list[Path]:
    raw = str(os.getenv("AIOPS_CODE_REPO_ROOTS", "")).strip()
    roots = _split_roots(raw) if raw else [str(_default_code_root())]
    resolved: list[Path] = []
    for item in roots:
        path = Path(item).expanduser()
        if path.exists():
            resolved.append(path)
    return resolved


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in code_roots():
        if root.is_file() and root.suffix.lower() in _SOURCE_SUFFIXES:
            files.append(root)
            continue
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
            )
    return sorted(files, key=lambda item: str(item))


def _iter_java_files() -> list[Path]:
    return [path for path in _iter_source_files() if path.suffix == ".java"]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, int(offset))) + 1


def _package_name(text: str) -> str:
    match = _PACKAGE_PATTERN.search(text)
    return str(match.group(1) or "").strip() if match else ""


def _full_name(package_name: str, class_name: str) -> str:
    return f"{package_name}.{class_name}" if package_name else class_name


def _class_for_offset(symbols: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    current: dict[str, Any] = {}
    for symbol in symbols:
        if int(symbol.get("offset") or 0) <= offset:
            current = symbol
        else:
            break
    return current


def _method_end_line(text: str, start_offset: int, start_line: int) -> int:
    depth = 0
    seen_open = False
    line_no = start_line
    for line_no, line in enumerate(text[start_offset:].splitlines(), start=start_line):
        depth += line.count("{")
        if "{" in line:
            seen_open = True
        depth -= line.count("}")
        if seen_open and depth <= 0:
            return line_no
    return line_no


def _symbols_in_file(path: Path, text: str | None = None) -> list[dict[str, Any]]:
    content = _read_text(path) if text is None else text
    package_name = _package_name(content)
    rows: list[dict[str, Any]] = []
    for match in _SYMBOL_PATTERN.finditer(content):
        name = str(match.group("name") or "").strip()
        kind = str(match.group("kind") or "").strip()
        line = _line_no(content, match.start())
        rows.append(
            {
                "symbolName": name,
                "className": name,
                "fullClassName": _full_name(package_name, name),
                "packageName": package_name,
                "kind": kind,
                "filePath": str(path),
                "line": line,
                "offset": match.start(),
            }
        )
    return rows


def _methods_in_file(path: Path, text: str | None = None) -> list[dict[str, Any]]:
    content = _read_text(path) if text is None else text
    package_name = _package_name(content)
    symbols = _symbols_in_file(path, content)
    rows: list[dict[str, Any]] = []
    for match in _METHOD_PATTERN.finditer(content):
        method_name = str(match.group("name") or "").strip()
        return_type = str(match.group("return") or "").strip()
        if method_name in {"if", "for", "while", "switch", "catch"}:
            continue
        symbol = _class_for_offset(symbols, match.start())
        class_name = str(symbol.get("className") or path.stem)
        start_line = _line_no(content, match.start())
        end_line = _method_end_line(content, match.start(), start_line)
        rows.append(
            {
                "className": class_name,
                "fullClassName": _full_name(package_name, class_name),
                "methodName": method_name,
                "signature": f"{method_name}(...)",
                "returnType": return_type,
                "filePath": str(path),
                "startLine": start_line,
                "endLine": end_line,
            }
        )
    return rows


def _candidate_symbol_names(text: str, extra_keywords: list[str] | None = None) -> list[str]:
    rows = [str(text or ""), *[str(item or "") for item in list(extra_keywords or [])]]
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for token in _TOKEN_PATTERN.findall(row):
            candidates = [token]
            if token and token[0].islower():
                candidates.append(f"{token[0].upper()}{token[1:]}")
            for candidate in candidates:
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                names.append(candidate)
    return names


def _business_expansion_terms(text: str, extra_keywords: list[str] | None = None) -> list[str]:
    rows = [str(text or ""), *[str(item or "") for item in list(extra_keywords or [])]]
    terms: list[str] = []
    seen: set[str] = set()
    for row in rows:
        lowered = str(row or "").lower()
        for trigger, expansions in _BUSINESS_TERM_EXPANSIONS.items():
            if trigger.lower() not in lowered:
                continue
            for item in expansions:
                value = str(item or "").strip()
                key = value.lower()
                if not value or key in seen:
                    continue
                seen.add(key)
                terms.append(value)
    return terms


def _candidate_text_terms(text: str, extra_keywords: list[str] | None = None) -> list[str]:
    rows = [
        str(text or ""),
        *[str(item or "") for item in list(extra_keywords or [])],
        *_business_expansion_terms(text, extra_keywords=extra_keywords),
    ]
    terms: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        item = str(term or "").strip()
        if len(item) < 2:
            return
        key = item.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(item)

    for row in rows:
        for token in _TOKEN_PATTERN.findall(row):
            _add(token)
        for phrase in _CJK_PATTERN.findall(row):
            _add(phrase)
            normalized = phrase
            for stop in _CJK_QUERY_STOP_PHRASES:
                normalized = normalized.replace(stop, " ")
            chunks = _CJK_PATTERN.findall(normalized)
            for chunk in chunks:
                _add(chunk)
            compact = "".join(chunks)
            if compact:
                _add(compact)
            for base in [compact, *chunks]:
                if len(base) < 3:
                    continue
                max_size = min(8, len(base))
                for size in range(max_size, 1, -1):
                    for start in range(0, len(base) - size + 1):
                        _add(base[start : start + size])

    terms.sort(key=lambda item: (len(item), item), reverse=True)
    return terms


def _line_text(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if line_no <= 0 or line_no > len(lines):
        return ""
    return lines[line_no - 1].strip()


def _next_symbol_for_offset(symbols: list[dict[str, Any]], offset: int) -> dict[str, Any]:
    for symbol in symbols:
        if int(symbol.get("offset") or 0) >= offset:
            return symbol
    return _class_for_offset(symbols, offset)


def _find_qflow_component_impls(component_ids: list[str]) -> list[dict[str, Any]]:
    wanted = {str(item or "").strip() for item in list(component_ids or []) if str(item or "").strip()}
    if not wanted:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in _iter_java_files():
        content = _read_text(path)
        if not any(component_id in content for component_id in wanted):
            continue
        symbols = _symbols_in_file(path, content)
        for match in _QFLOW_COMPONENT_ANNOTATION_PATTERN.finditer(content):
            component_id = str(match.group("component_id") or "").strip()
            if component_id not in wanted:
                continue
            key = (component_id, str(path))
            if key in seen:
                continue
            seen.add(key)
            symbol = dict(_next_symbol_for_offset(symbols, match.start()))
            symbol.pop("offset", None)
            symbol["componentId"] = component_id
            line_no = _line_no(content, match.start())
            rows.append(
                {
                    "filePath": str(path),
                    "line": line_no,
                    "text": _line_text(content, line_no),
                    "matched": [component_id],
                    "score": 0,
                    "kind": "qflow_component_impl",
                    "componentId": component_id,
                    "symbol": symbol,
                }
            )
    return rows


def _methods_for_symbol(symbol: dict[str, Any]) -> list[dict[str, Any]]:
    file_path = str(dict(symbol or {}).get("filePath") or "").strip()
    class_name = str(dict(symbol or {}).get("className") or "").strip()
    if not file_path or not class_name:
        return []
    path = Path(file_path)
    if not path.exists() or path.suffix != ".java":
        return []
    methods = [
        dict(item or {})
        for item in _methods_in_file(path)
        if str(dict(item or {}).get("className") or "") == class_name
    ]
    return methods


def _preferred_entry_method(methods: list[dict[str, Any]]) -> dict[str, Any]:
    if not methods:
        return {}
    for name in _ENTRY_METHOD_NAMES:
        for method in methods:
            if str(method.get("methodName") or "") == name:
                return dict(method)
    return dict(methods[0] or {})


def search_symbol(name: str) -> dict[str, Any]:
    target = str(name or "").strip()
    if not target:
        return {"ok": False, "symbol": {}, "error": "empty symbol"}
    lowered = target.lower()
    for path in _iter_java_files():
        content = _read_text(path)
        for symbol in _symbols_in_file(path, content):
            symbol_name = str(symbol.get("symbolName") or "")
            if symbol_name.lower() == lowered or path.stem.lower() == lowered:
                clean = dict(symbol)
                clean.pop("offset", None)
                return {"ok": True, "symbol": clean, "error": ""}
    return {"ok": False, "symbol": {}, "error": f"symbol not found: {target}"}


def search_method_local(name: str) -> dict[str, Any]:
    target = str(name or "").strip()
    if not target:
        return {"ok": False, "methods": [], "error": "empty method"}
    lowered = target.lower()
    rows: list[dict[str, Any]] = []
    for path in _iter_java_files():
        for method in _methods_in_file(path):
            method_name = str(method.get("methodName") or "")
            if method_name.lower() == lowered:
                rows.append(method)
    if not rows:
        return {"ok": False, "methods": [], "error": f"method not found: {target}"}
    rows.sort(key=lambda item: (str(item.get("filePath") or ""), int(item.get("startLine") or 0)))
    return {"ok": True, "methods": rows[:10], "error": ""}


def locate_line_local(class_or_file: str, line: int) -> dict[str, Any]:
    target = str(class_or_file or "").strip()
    line_no = int(line or 0)
    if not target or line_no <= 0:
        return {"ok": False, "result": {}, "error": "invalid class_or_file/line"}
    normalized = target[:-5] if target.endswith(".java") else target
    lowered = normalized.lower()
    for path in _iter_java_files():
        if path.stem.lower() != lowered and target.lower() not in str(path).lower():
            continue
        content = _read_text(path)
        symbols = _symbols_in_file(path, content)
        current_symbol: dict[str, Any] = {}
        for symbol in symbols:
            if int(symbol.get("line") or 0) <= line_no:
                current_symbol = dict(symbol)
            else:
                break
        for method in _methods_in_file(path, content):
            start_line = int(method.get("startLine") or 0)
            end_line = int(method.get("endLine") or 0)
            if start_line <= line_no <= end_line:
                clean_symbol = dict(current_symbol)
                clean_symbol.pop("offset", None)
                return {
                    "ok": True,
                    "result": {
                        "method": method,
                        "symbol": clean_symbol,
                        "line": {
                            "filePath": str(path),
                            "line": line_no,
                            "text": _line_text(content, line_no),
                        },
                    },
                    "error": "",
                }
        clean_symbol = dict(current_symbol)
        clean_symbol.pop("offset", None)
        return {
            "ok": True,
            "result": {
                "method": {},
                "symbol": clean_symbol,
                "line": {
                    "filePath": str(path),
                    "line": line_no,
                    "text": _line_text(content, line_no),
                },
            },
            "error": "",
        }
    return {"ok": False, "result": {}, "error": f"line target not found: {target}:{line_no}"}


def search_text(query: str) -> dict[str, Any]:
    candidates = _candidate_text_terms(query)
    if not candidates:
        return {"ok": False, "matches": [], "error": "empty query"}
    matches: list[dict[str, Any]] = []
    lowered_candidates = [item.lower() for item in candidates]
    for path in _iter_source_files():
        content = _read_text(path)
        for line_no, line in enumerate(content.splitlines(), start=1):
            lowered_line = line.lower()
            matched = [item for item in lowered_candidates if item in lowered_line]
            if not matched:
                continue
            score = sum(len(item) for item in matched)
            path_text = str(path)
            if path.suffix.lower() == ".xml":
                score += 20
            if "/resources/flow/" in path_text:
                score += 40
            if "desc=" in line or "qflow:" in line:
                score += 20
            matches.append(
                {
                    "filePath": str(path),
                    "line": line_no,
                    "text": line.strip(),
                    "matched": matched,
                    "score": score,
                }
            )
    if not matches:
        return {"ok": False, "matches": [], "error": "text not found"}
    component_ids: list[str] = []
    seen_component_ids: set[str] = set()
    max_xml_score_by_id: dict[str, int] = {}
    for match in matches:
        if str(match.get("filePath") or "").lower().endswith(".xml"):
            for component_id in _QFLOW_XML_COMPONENT_ID_PATTERN.findall(str(match.get("text") or "")):
                component_id = str(component_id or "").strip()
                if not component_id:
                    continue
                if component_id not in seen_component_ids:
                    seen_component_ids.add(component_id)
                    component_ids.append(component_id)
                max_xml_score_by_id[component_id] = max(
                    int(match.get("score") or 0),
                    int(max_xml_score_by_id.get(component_id) or 0),
                )
    impl_matches = _find_qflow_component_impls(component_ids)
    for impl in impl_matches:
        component_id = str(impl.get("componentId") or "").strip()
        impl["score"] = max(1, int(max_xml_score_by_id.get(component_id) or 1) - 1)
    matches.extend(impl_matches)
    matches.sort(
        key=lambda item: (
            int(item.get("score") or 0),
            -int(item.get("line") or 0),
            str(item.get("filePath") or ""),
        ),
        reverse=True,
    )
    return {"ok": True, "matches": matches[:_MAX_TEXT_MATCHES], "error": ""}


def _is_business_entry_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    return any(term in text for term in _BUSINESS_ENTRY_TERMS)


def search_business_entry(question: str) -> dict[str, Any]:
    if not _is_business_entry_question(question):
        return {"ok": False, "methods": [], "error": "not a business-entry question"}
    candidates: list[dict[str, Any]] = []
    for path in _iter_java_files():
        path_text = str(path)
        if "CreateOrder" not in path_text and "create" not in path_text.lower():
            continue
        for method in _methods_in_file(path):
            if str(method.get("methodName") or "") != "createOrder":
                continue
            class_name = str(method.get("className") or "")
            score = 0
            if class_name in {"SingleCreateOrderServiceImpl", "DoubleCreateOrderServiceImpl"}:
                score += 100
            if "CreateOrderService" in class_name:
                score += 50
            if "test/" in str(method.get("filePath") or ""):
                score -= 30
            row = dict(method)
            row["score"] = score
            candidates.append(row)
    candidates.sort(
        key=lambda item: (
            int(item.get("score") or 0),
            1 if str(item.get("className") or "") == "SingleCreateOrderServiceImpl" else 0,
            str(item.get("className") or ""),
        ),
        reverse=True,
    )
    if not candidates:
        return {"ok": False, "methods": [], "error": "business entry not found"}
    return {"ok": True, "methods": candidates[:5], "error": ""}


def analyze_local_code(
    *,
    question: str,
    extra_keywords: list[str] | None = None,
) -> dict[str, Any]:
    for symbol_name in _candidate_symbol_names(question, extra_keywords=extra_keywords):
        found = search_symbol(symbol_name)
        if found.get("ok"):
            symbol = dict(found.get("symbol") or {})
            summary = (
                f"本地源码命中 {symbol.get('kind')} {symbol.get('fullClassName')} "
                f"{symbol.get('filePath')}:{symbol.get('line')}"
            )
            return {
                "ok": True,
                "mode": "local_symbol",
                "summary": summary,
                "current_method": {},
                "current_symbol": symbol,
                "caller": [],
                "callee": [],
                "logs": [],
                "matched_methods": [],
                "evidence": [
                    f"[local_code_index] symbol={symbol.get('symbolName')}",
                    f"[local_code_index] {summary}",
                ],
                "error": "",
            }

    entry = search_business_entry(question)
    if entry.get("ok"):
        methods = [dict(item or {}) for item in list(entry.get("methods") or [])]
        top = dict(methods[0] or {})
        summary = (
            f"本地源码命中业务入口 {top.get('className')}.{top.get('methodName')} "
            f"{top.get('filePath')}:{top.get('startLine')}"
        )
        return {
            "ok": True,
            "mode": "local_business_entry",
            "summary": summary,
            "current_method": top,
            "current_symbol": {},
            "caller": [],
            "callee": [],
            "logs": [],
            "matched_methods": methods,
            "evidence": [
                "[local_code_index] business_terms=生单/createOrder",
                f"[local_code_index] {summary}",
            ],
            "error": "",
        }

    for method_name in _candidate_symbol_names(question, extra_keywords=extra_keywords):
        found_methods = search_method_local(method_name)
        if found_methods.get("ok"):
            methods = [dict(item or {}) for item in list(found_methods.get("methods") or [])]
            top = dict(methods[0] or {})
            summary = (
                f"本地源码命中方法 {top.get('className')}.{top.get('methodName')} "
                f"{top.get('filePath')}:{top.get('startLine')}"
            )
            return {
                "ok": True,
                "mode": "local_method",
                "summary": summary,
                "current_method": top,
                "current_symbol": {},
                "caller": [],
                "callee": [],
                "logs": [],
                "matched_methods": methods,
                "text_matches": [],
                "evidence": [
                    f"[local_code_index] method={top.get('methodName')}",
                    f"[local_code_index] {summary}",
                ],
                "error": "",
            }

    found_text = search_text(question)
    if found_text.get("ok"):
        matches = [dict(item or {}) for item in list(found_text.get("matches") or [])]
        top = dict(matches[0] or {})
        impl_matches = [
            dict(item or {})
            for item in matches
            if str(dict(item or {}).get("kind") or "") == "qflow_component_impl"
        ]
        impl_symbol = dict(dict(impl_matches[0] or {}).get("symbol") or {}) if impl_matches else {}
        impl_methods = _methods_for_symbol(impl_symbol) if impl_symbol else []
        impl_entry_method = _preferred_entry_method(impl_methods)
        summary = f"本地源码全文命中 {top.get('filePath')}:{top.get('line')} {top.get('text')}"
        if impl_symbol:
            summary = (
                f"{summary}; qflow组件 {impl_symbol.get('componentId')} 实现类 "
                f"{impl_symbol.get('fullClassName')} {impl_symbol.get('filePath')}:{impl_symbol.get('line')}"
            )
        if impl_entry_method:
            summary = (
                f"{summary}; 入口方法 {impl_entry_method.get('className')}."
                f"{impl_entry_method.get('methodName')}:{impl_entry_method.get('startLine')}"
            )
        evidence = [
            f"[local_code_index] text={top.get('filePath')}:{top.get('line')}",
            f"[local_code_index] {summary}",
        ]
        for impl in impl_matches[:2]:
            symbol = dict(impl.get("symbol") or {})
            evidence.append(
                f"[local_code_index] qflow_component id={impl.get('componentId')} "
                f"class={symbol.get('fullClassName')} file={impl.get('filePath')}:{impl.get('line')}"
            )
        if impl_entry_method:
            evidence.append(
                f"[local_code_index] entry_method class={impl_entry_method.get('className')} "
                f"method={impl_entry_method.get('methodName')} "
                f"file={impl_entry_method.get('filePath')}:{impl_entry_method.get('startLine')}"
            )
        return {
            "ok": True,
            "mode": "local_text",
            "summary": summary,
            "current_method": impl_entry_method,
            "current_symbol": impl_symbol,
            "caller": [],
            "callee": [],
            "logs": [],
            "matched_methods": impl_methods,
            "text_matches": matches,
            "evidence": evidence,
            "error": "",
        }

    return {
        "ok": False,
        "mode": "none",
        "summary": "local_code_index 未命中可用代码上下文",
        "current_method": {},
        "current_symbol": {},
        "caller": [],
        "callee": [],
        "logs": [],
        "matched_methods": [],
        "evidence": [],
        "error": "no local source result",
    }
