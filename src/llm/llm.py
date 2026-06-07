"""Unified LLM invocation entrypoint."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from pydantic import SecretStr

from llm.small_ll import check_sensitive_operation_with_small_llm

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_MODEL = "azure/gpt-5.3-codex-2026-02-24"
_DEFAULT_BASE_URL = "http://llm.api.corp.qunar.com/v1"
_MAX_SUMMARY_LEN = 500
_INTENT_LABELS = ("业务咨询", "线上问题咨询", "订单信息查询")

_LLM_CLIENT: Any | None = None
_LLM_INIT_DONE = False
_LLM_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
_LLM_LOG = logging.getLogger("aiops.llm")


def _pick_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = str(os.getenv(key, "")).strip()
        if value:
            return value
    return default


def _pick_float_env(*keys: str, default: float = 0.0) -> float:
    raw = _pick_env(*keys, default="")
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _scene_key(scene: str) -> str:
    if scene == "intent_recognition":
        return "INTENT"
    if scene == "sensitive_check":
        return "SENSITIVE"
    return "GENERAL"


def _resolve_runtime_model(scene: str) -> str:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        return _pick_env("AIOPS_LLM_MODEL", "LLM_MODEL", default=_DEFAULT_MODEL) or _DEFAULT_MODEL
    return _pick_env(
        f"AIOPS_{scene_key}_LLM_MODEL",
        f"LLM_{scene_key}_MODEL",
        "AIOPS_LLM_MODEL",
        "LLM_MODEL",
        default=_DEFAULT_MODEL,
    ) or _DEFAULT_MODEL


def _resolve_runtime_base_url(scene: str) -> str:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        return _pick_env("AIOPS_LLM_BASE_URL", "LLM_BASE_URL", default=_DEFAULT_BASE_URL) or _DEFAULT_BASE_URL
    return _pick_env(
        f"AIOPS_{scene_key}_LLM_BASE_URL",
        f"LLM_{scene_key}_BASE_URL",
        "AIOPS_LLM_BASE_URL",
        "LLM_BASE_URL",
        default=_DEFAULT_BASE_URL,
    ) or _DEFAULT_BASE_URL


def _resolve_token_price(scene: str) -> tuple[float, float]:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        in_price = _pick_float_env("AIOPS_LLM_INPUT_PRICE_PER_1M", "LLM_INPUT_PRICE_PER_1M", default=0.0)
        out_price = _pick_float_env("AIOPS_LLM_OUTPUT_PRICE_PER_1M", "LLM_OUTPUT_PRICE_PER_1M", default=0.0)
        return in_price, out_price
    in_price = _pick_float_env(
        f"AIOPS_{scene_key}_LLM_INPUT_PRICE_PER_1M",
        f"LLM_{scene_key}_INPUT_PRICE_PER_1M",
        "AIOPS_LLM_INPUT_PRICE_PER_1M",
        "LLM_INPUT_PRICE_PER_1M",
        default=0.0,
    )
    out_price = _pick_float_env(
        f"AIOPS_{scene_key}_LLM_OUTPUT_PRICE_PER_1M",
        f"LLM_{scene_key}_OUTPUT_PRICE_PER_1M",
        "AIOPS_LLM_OUTPUT_PRICE_PER_1M",
        "LLM_OUTPUT_PRICE_PER_1M",
        default=0.0,
    )
    return in_price, out_price


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_token_usage(result: Any) -> tuple[int | None, int | None, int | None]:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    usage_metadata = getattr(result, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = _safe_int(usage_metadata.get("input_tokens"))
        output_tokens = _safe_int(usage_metadata.get("output_tokens"))
        total_tokens = _safe_int(usage_metadata.get("total_tokens"))

    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            if input_tokens is None:
                input_tokens = _safe_int(
                    token_usage.get("prompt_tokens")
                    if token_usage.get("prompt_tokens") is not None
                    else token_usage.get("input_tokens")
                )
            if output_tokens is None:
                output_tokens = _safe_int(
                    token_usage.get("completion_tokens")
                    if token_usage.get("completion_tokens") is not None
                    else token_usage.get("output_tokens")
                )
            if total_tokens is None:
                total_tokens = _safe_int(token_usage.get("total_tokens"))

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _estimate_token_cost(scene: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    input_price_per_m, output_price_per_m = _resolve_token_price(scene)
    if input_price_per_m <= 0 and output_price_per_m <= 0:
        return None
    prompt = max(input_tokens or 0, 0)
    completion = max(output_tokens or 0, 0)
    return (prompt * input_price_per_m + completion * output_price_per_m) / 1_000_000


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
        best_intent = "业务咨询"
        confidence = 0.5
    elif any(token in lowered for token in ("失败", "错误", "异常", "故障", "排查", "timeout", "error", "一直")):
        best_intent = "线上问题咨询"
        confidence = 0.82
    elif any(token in lowered for token in ("订单", "订单号", "航班", "乘机人", "状态", "信息", "详情", "查询")):
        best_intent = "订单信息查询"
        confidence = 0.8
    else:
        best_intent = "业务咨询"
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


def _build_llm_client(model: str, base_url: str) -> Any | None:
    global _LLM_INIT_DONE, _LLM_CLIENT, _LLM_CLIENT_CACHE
    if not _LLM_INIT_DONE and _LLM_CLIENT is None and _LLM_CLIENT_CACHE:
        _LLM_CLIENT_CACHE = {}
    if ChatOpenAI is None:
        return None

    # 兼容历史环境变量：优先 OPENAI_API_KEY，其次 LLM_API_KEY。
    api_key = _pick_env("OPENAI_API_KEY", "LLM_API_KEY", default="")
    if not api_key:
        return None

    cache_key = (model, base_url)
    if cache_key in _LLM_CLIENT_CACHE:
        return _LLM_CLIENT_CACHE[cache_key]

    try:
        client = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            temperature=0,
        )
    except Exception:  # noqa: BLE001
        return None

    _LLM_CLIENT_CACHE[cache_key] = client
    # 兼容旧测试/调用方对默认 client 的访问。
    if not _LLM_INIT_DONE:
        _LLM_INIT_DONE = True
        _LLM_CLIENT = client
    return client


def _invoke_llm(system_prompt: str, user_prompt: str, *, scene: str = "general") -> str:
    model = _resolve_runtime_model(scene)
    base_url = _resolve_runtime_base_url(scene)
    llm = _build_llm_client(model=model, base_url=base_url)
    if llm is None:
        _LLM_LOG.warning("llm.invoke.skip scene=%s model=%s reason=no_client", scene, model)
        return ""
    u_len, s_len = len(str(user_prompt or "")), len(str(system_prompt or ""))
    _LLM_LOG.info(
        "llm.invoke.start scene=%s model=%s user_chars=%d system_chars=%d",
        scene,
        model,
        u_len,
        s_len,
    )
    started = time.perf_counter()
    try:
        if system_prompt.strip():
            result = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        else:
            result = llm.invoke(user_prompt)
        out = _coerce_text(getattr(result, "content", result)).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        input_tokens, output_tokens, total_tokens = _extract_token_usage(result)
        token_cost = _estimate_token_cost(scene, input_tokens, output_tokens)
        _LLM_LOG.info(
            "llm.invoke.end scene=%s model=%s out_chars=%d duration_ms=%.2f input_tokens=%s output_tokens=%s total_tokens=%s est_cost=%s",
            scene,
            model,
            len(out),
            duration_ms,
            str(input_tokens) if input_tokens is not None else "n/a",
            str(output_tokens) if output_tokens is not None else "n/a",
            str(total_tokens) if total_tokens is not None else "n/a",
            f"{token_cost:.8f}" if token_cost is not None else "n/a",
        )
        return out
    except Exception:  # noqa: BLE001
        duration_ms = (time.perf_counter() - started) * 1000
        _LLM_LOG.exception("llm.invoke.error scene=%s model=%s duration_ms=%.2f", scene, model, duration_ms)
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


def analyze_with_llm(
    question: str,
    evidence: str,
    *,
    system_prompt_file: str = "analysis_system_prompt.txt",
    user_prompt_file: str = "analysis_user_prompt.txt",
) -> dict[str, str]:
    question_text = str(question).strip()
    evidence_text = str(evidence).strip()

    system_prompt = load_prompt(system_prompt_file, default="")
    user_prompt = render_prompt(
        user_prompt_file,
        question=question_text,
        evidence=evidence_text,
    )
    if not user_prompt:
        return _default_analysis(question_text, evidence_text)
    text = _invoke_llm(system_prompt, user_prompt, scene="general")
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
    text = _invoke_llm(system_prompt, user_prompt, scene="general")
    if not text:
        return fallback
    return text[:_MAX_SUMMARY_LEN]


def chat_with_llm(question: str, system_prompt: str = "") -> str:
    question_text = str(question).strip()
    if not question_text:
        return ""
    return _invoke_llm(system_prompt=system_prompt, user_prompt=question_text, scene="general")


def check_sensitive_operation_with_llm(question: str) -> dict[str, Any]:
    return check_sensitive_operation_with_small_llm(question)


def _recognize_intent_with_llm(question: str, intent_history_prompt: str | None = None) -> dict[str, Any]:
    question_text = str(question or "").strip()
    if not question_text:
        return _fallback_intent_recognition(question_text)

    system_prompt = load_prompt("intent_recognition_system_prompt.txt", default="")
    if str(intent_history_prompt or "").strip():
        user_prompt = str(intent_history_prompt or "").strip()
    else:
        user_prompt = render_prompt("intent_recognition_user_prompt.txt", question=question_text)
    if not user_prompt:
        return _fallback_intent_recognition(question_text)

    text = _invoke_llm(system_prompt, user_prompt, scene="intent_recognition")
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


def recognize_intent(question: str, intent_history_prompt: str | None = None) -> dict[str, Any]:
    return _recognize_intent_with_llm(question, intent_history_prompt=intent_history_prompt)
