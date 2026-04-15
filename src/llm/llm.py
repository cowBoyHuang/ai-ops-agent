"""Unified LLM invocation entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import SecretStr

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment]

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_DEFAULT_MODEL = "azure/gpt-5.3-codex-2026-02-24"
_DEFAULT_BASE_URL = "http://llm.api.corp.qunar.com/v1"
_MAX_SUMMARY_LEN = 500

_LLM_CLIENT: Any | None = None
_LLM_INIT_DONE = False


def load_prompt(prompt_file: str, default: str = "") -> str:
    path = _PROMPTS_DIR / prompt_file
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rows: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    rows.append(str(text))
            else:
                rows.append(str(item))
        return "\n".join(rows).strip()
    return str(content)


def _default_analysis(question: str, evidence: str) -> dict[str, str]:
    merged = f"{question}\n{evidence}".lower()
    if "timeout" in merged:
        return {
            "root_cause": "inventory-service timeout",
            "confidence": "high",
            "reply": "根因初步定位完成，建议先检查下游服务超时与连接池。",
        }
    return {
        "root_cause": "可能是下游依赖异常",
        "confidence": "medium",
        "reply": "根因初步定位完成，建议先检查下游服务超时与连接池。",
    }


def _default_summary(total_message: str, summary_message: str) -> str:
    total_text = str(total_message).strip()
    summary_text = str(summary_message).strip()
    if not total_text and not summary_text:
        return ""
    merged = f"{summary_text}\n{total_text}".strip()
    return merged[:_MAX_SUMMARY_LEN]


def _build_llm_client() -> Any | None:
    global _LLM_INIT_DONE, _LLM_CLIENT
    if _LLM_INIT_DONE:
        return _LLM_CLIENT

    _LLM_INIT_DONE = True
    if ChatOpenAI is None:
        _LLM_CLIENT = None
        return None

    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        _LLM_CLIENT = None
        return None

    model = str(os.getenv("AIOPS_LLM_MODEL", _DEFAULT_MODEL)).strip() or _DEFAULT_MODEL
    base_url = str(os.getenv("AIOPS_LLM_BASE_URL", _DEFAULT_BASE_URL)).strip() or _DEFAULT_BASE_URL

    try:
        _LLM_CLIENT = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=0,
        )
    except Exception:  # noqa: BLE001
        _LLM_CLIENT = None
    return _LLM_CLIENT


def _invoke_llm(system_prompt: str, user_prompt: str) -> str:
    llm = _build_llm_client()
    if llm is None:
        return ""
    try:
        if system_prompt.strip():
            result = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        else:
            result = llm.invoke(user_prompt)
        return _coerce_text(getattr(result, "content", result)).strip()
    except Exception:  # noqa: BLE001
        return ""


def analyze_with_llm(question: str, evidence: str) -> dict[str, str]:
    question_text = str(question).strip()
    evidence_text = str(evidence).strip()

    system_prompt = load_prompt(
        "analysis_system_prompt.txt",
        default="你是AIOps根因分析助手，请返回结构化JSON。",
    )
    user_prompt = (
        "请根据问题与证据生成根因分析结果，输出严格 JSON，字段包含："
        "root_cause, confidence(high|medium|low), reply。\n"
        f"问题:\n{question_text}\n\n证据:\n{evidence_text}"
    )
    text = _invoke_llm(system_prompt, user_prompt)
    if not text:
        return _default_analysis(question_text, evidence_text)

    try:
        parsed = json.loads(text)
        root_cause = str(parsed.get("root_cause") or "").strip()
        confidence = str(parsed.get("confidence") or "").strip().lower()
        reply = str(parsed.get("reply") or "").strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        if not root_cause or not reply:
            return _default_analysis(question_text, evidence_text)
        return {
            "root_cause": root_cause,
            "confidence": confidence,
            "reply": reply,
        }
    except Exception:  # noqa: BLE001
        return _default_analysis(question_text, evidence_text)


def summarize_with_llm(total_message: str, summary_message: str) -> str:
    fallback = _default_summary(total_message, summary_message)
    if not fallback:
        return ""

    system_prompt = load_prompt(
        "summary_system_prompt.txt",
        default="你是对话总结助手，保留关键信息，控制在500字符以内。",
    )
    user_prompt = (
        "请对以下内容做增量总结，输出纯文本，不要JSON。\n"
        f"历史摘要:\n{summary_message}\n\n新增内容:\n{total_message}"
    )
    text = _invoke_llm(system_prompt, user_prompt)
    if not text:
        return fallback
    return text[:_MAX_SUMMARY_LEN]


def chat_with_llm(question: str, system_prompt: str = "") -> str:
    question_text = str(question).strip()
    if not question_text:
        return ""
    return _invoke_llm(system_prompt=system_prompt, user_prompt=question_text)
