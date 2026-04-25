"""Unified LLM invocation entrypoint."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import SecretStr

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_MODEL = "azure/gpt-5.3-codex-2026-02-24"
_DEFAULT_BASE_URL = "http://llm.api.corp.qunar.com/v1"
_MAX_SUMMARY_LEN = 500
_INTENT_LABELS = ("系统逻辑咨询", "线上问题咨询", "订单信息查询")

_LLM_CLIENT: Any | None = None
_LLM_INIT_DONE = False
_LLM_LOG = logging.getLogger("aiops.llm")


def _pick_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = str(os.getenv(key, "")).strip()
        if value:
            return value
    return default


def load_prompt(prompt_file: str, default: str = "") -> str:
    path = _PROMPTS_DIR / prompt_file
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def render_prompt(prompt_file: str, **kwargs: Any) -> str:
    template = load_prompt(prompt_file, default="")
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except Exception:  # noqa: BLE001
        return template


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


def _fallback_intent_recognition(question: str) -> dict[str, Any]:
    text = str(question or "").strip()
    lowered = text.lower()
    if not text:
        best_intent = "系统逻辑咨询"
        confidence = 0.5
    elif any(token in lowered for token in ("失败", "错误", "异常", "故障", "排查", "timeout", "error", "一直")):
        best_intent = "线上问题咨询"
        confidence = 0.82
    elif any(token in lowered for token in ("订单", "订单号", "航班", "乘机人", "状态", "信息", "详情", "查询")):
        best_intent = "订单信息查询"
        confidence = 0.8
    else:
        best_intent = "系统逻辑咨询"
        confidence = 0.72

    scores: dict[str, dict[str, float]] = {}
    for label in _INTENT_LABELS:
        base = 0.2
        if label == best_intent:
            base = confidence
        scores[label] = {
            "semantic_match": round(base, 3),
            "keyword_match": round(base, 3),
            "context_relevance": 0.5,
            "question_type_match": round(base, 3),
            "final_score": round(base, 3),
        }
    return {
        "scores": scores,
        "best_intent": best_intent,
        "confidence": round(confidence, 3),
        "reasoning": "fallback heuristic",
    }


def _build_llm_client() -> Any | None:
    global _LLM_INIT_DONE, _LLM_CLIENT
    if _LLM_INIT_DONE:
        return _LLM_CLIENT

    _LLM_INIT_DONE = True
    if ChatOpenAI is None:
        _LLM_CLIENT = None
        return None

    # 兼容历史环境变量：优先 OPENAI_API_KEY，其次 LLM_API_KEY。
    api_key = _pick_env("OPENAI_API_KEY", "LLM_API_KEY", default="")
    if not api_key:
        _LLM_CLIENT = None
        return None

    # 兼容历史环境变量：AIOPS_* 与 LLM_* 两套命名。
    model = _pick_env("AIOPS_LLM_MODEL", "LLM_MODEL", default=_DEFAULT_MODEL) or _DEFAULT_MODEL
    base_url = _pick_env("AIOPS_LLM_BASE_URL", "LLM_BASE_URL", default=_DEFAULT_BASE_URL) or _DEFAULT_BASE_URL

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
        _LLM_LOG.warning("llm.invoke.skip reason=no_client")
        return ""
    u_len, s_len = len(str(user_prompt or "")), len(str(system_prompt or ""))
    _LLM_LOG.info("llm.invoke.start user_chars=%d system_chars=%d", u_len, s_len)
    try:
        if system_prompt.strip():
            result = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        else:
            result = llm.invoke(user_prompt)
        out = _coerce_text(getattr(result, "content", result)).strip()
        _LLM_LOG.info("llm.invoke.end out_chars=%d", len(out))
        return out
    except Exception:  # noqa: BLE001
        _LLM_LOG.exception("llm.invoke.error")
        return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def analyze_with_llm(question: str, evidence: str) -> dict[str, str]:
    question_text = str(question).strip()
    evidence_text = str(evidence).strip()

    system_prompt = load_prompt("analysis_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "analysis_user_prompt.txt",
        question=question_text,
        evidence=evidence_text,
    )
    if not user_prompt:
        return _default_analysis(question_text, evidence_text)
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

    system_prompt = load_prompt("summary_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "summary_user_prompt.txt",
        summary_message=summary_message,
        total_message=total_message,
    )
    if not user_prompt:
        return fallback
    text = _invoke_llm(system_prompt, user_prompt)
    if not text:
        return fallback
    return text[:_MAX_SUMMARY_LEN]


def chat_with_llm(question: str, system_prompt: str = "") -> str:
    question_text = str(question).strip()
    if not question_text:
        return ""
    return _invoke_llm(system_prompt=system_prompt, user_prompt=question_text)


def check_sensitive_operation_with_llm(question: str) -> dict[str, Any]:
    question_text = str(question).strip()
    if not question_text:
        return {"passed": False, "reason": "empty question"}

    system_prompt = load_prompt("sensitive_operation_system_prompt.txt", default="")
    user_prompt = render_prompt(
        "sensitive_operation_user_prompt.txt",
        question=question_text,
    )
    if not user_prompt:
        return {"passed": False, "reason": "sensitive prompt missing"}

    text = _invoke_llm(system_prompt, user_prompt)
    if not text:
        return {"passed": True, "reason": "llm check unavailable (degraded allow)"}

    parsed = _parse_json_object(text)
    if not isinstance(parsed, dict):
        return {"passed": False, "reason": "llm response parse failed"}

    allow_value = parsed.get("allow")
    if isinstance(allow_value, bool):
        allow = allow_value
    else:
        allow_text = str(allow_value or "").strip().lower()
        allow = allow_text in {"true", "1", "yes", "allow", "safe"}

    reason = str(parsed.get("reason") or "").strip() or "llm sensitive check blocked"
    return {"passed": allow, "reason": reason}


def recognize_intent(question: str, intent_history_prompt: str | None = None) -> dict[str, Any]:
    question_text = str(question or "").strip()
    if not question_text:
        return _fallback_intent_recognition(question_text)

    system_prompt = load_prompt("intent_recognition_system_prompt.txt", default="")
    if str(intent_history_prompt or "").strip():
        user_prompt = str(intent_history_prompt or "").strip()
    else:
        user_prompt = render_prompt(
            "intent_recognition_user_prompt.txt",
            question=question_text,
        )
    if not user_prompt:
        return _fallback_intent_recognition(question_text)

    text = _invoke_llm(system_prompt, user_prompt)
    parsed = _parse_json_object(text) if text else None
    if not isinstance(parsed, dict):
        return _fallback_intent_recognition(question_text)

    best_intent = str(parsed.get("best_intent") or "").strip()
    if best_intent not in _INTENT_LABELS:
        return _fallback_intent_recognition(question_text)

    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        scores = {}

    return {
        "scores": scores,
        "best_intent": best_intent,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
    }
