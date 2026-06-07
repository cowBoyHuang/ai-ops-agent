"""Local small-LLM entrypoint (for on-prem Ollama models)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    import ollama
except Exception:  # noqa: BLE001
    ollama = None  # type: ignore[assignment]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_SMALL_MODEL = "intent-guard"
_DEFAULT_SMALL_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_GENERAL_MAX_TOKENS = 512
_DEFAULT_INTENT_MAX_TOKENS = 320
_DEFAULT_SENSITIVE_MAX_TOKENS = 96
_INTENT_LABELS = ("业务咨询", "线上问题咨询", "订单信息查询")

_SMALL_LLM_CLIENT: Any | None = None
_SMALL_LLM_INIT_DONE = False
_SMALL_LLM_CLIENT_CACHE: dict[str, Any] = {}
_SMALL_LLM_LOG = logging.getLogger("aiops.small_llm")


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


def _scene_key(scene: str) -> str:
    if scene == "intent_recognition":
        return "INTENT"
    if scene == "sensitive_check":
        return "SENSITIVE"
    return "GENERAL"


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # noqa: BLE001
            pass
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # noqa: BLE001
            pass
    return {}


def _pick_token_value(rows: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _safe_int(rows.get(key))
        if value is not None:
            return value
    return None


def _default_small_llm_max_tokens(scene: str) -> int:
    scene_key = _scene_key(scene)
    if scene_key == "INTENT":
        return _DEFAULT_INTENT_MAX_TOKENS
    if scene_key == "SENSITIVE":
        return _DEFAULT_SENSITIVE_MAX_TOKENS
    return _DEFAULT_GENERAL_MAX_TOKENS


def _extract_token_usage(result: Any) -> tuple[int | None, int | None, int | None]:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    rows = _coerce_dict(result)

    input_keys = ("prompt_eval_count", "input_tokens", "prompt_tokens", "prompt_token_count")
    output_keys = ("eval_count", "output_tokens", "completion_tokens", "completion_token_count")
    total_keys = ("total_tokens", "total_token_count")

    input_tokens = _pick_token_value(rows, input_keys)
    output_tokens = _pick_token_value(rows, output_keys)
    total_tokens = _pick_token_value(rows, total_keys)

    usage = _coerce_dict(rows.get("usage"))
    if input_tokens is None:
        input_tokens = _pick_token_value(usage, input_keys)
    if output_tokens is None:
        output_tokens = _pick_token_value(usage, output_keys)
    if total_tokens is None:
        total_tokens = _pick_token_value(usage, total_keys)

    usage_metadata = _coerce_dict(rows.get("usage_metadata"))
    if input_tokens is None:
        input_tokens = _pick_token_value(usage_metadata, input_keys)
    if output_tokens is None:
        output_tokens = _pick_token_value(usage_metadata, output_keys)
    if total_tokens is None:
        total_tokens = _pick_token_value(usage_metadata, total_keys)

    token_usage = _coerce_dict(rows.get("token_usage"))
    if input_tokens is None:
        input_tokens = _pick_token_value(token_usage, ("prompt_tokens", "input_tokens", "prompt_eval_count"))
    if output_tokens is None:
        output_tokens = _pick_token_value(token_usage, ("completion_tokens", "output_tokens", "eval_count"))
    if total_tokens is None:
        total_tokens = _pick_token_value(token_usage, total_keys)

    response_metadata = _coerce_dict(rows.get("response_metadata"))
    response_token_usage = _coerce_dict(response_metadata.get("token_usage"))
    if input_tokens is None:
        input_tokens = _pick_token_value(response_token_usage, ("prompt_tokens", "input_tokens", "prompt_eval_count"))
    if output_tokens is None:
        output_tokens = _pick_token_value(response_token_usage, ("completion_tokens", "output_tokens", "eval_count"))
    if total_tokens is None:
        total_tokens = _pick_token_value(response_token_usage, total_keys)

    if input_tokens is None:
        for key in input_keys:
            value = _safe_int(getattr(result, key, None))
            if value is not None:
                input_tokens = value
                break
    if output_tokens is None:
        for key in output_keys:
            value = _safe_int(getattr(result, key, None))
            if value is not None:
                output_tokens = value
                break
    if total_tokens is None:
        for key in total_keys:
            value = _safe_int(getattr(result, key, None))
            if value is not None:
                total_tokens = value
                break

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _resolve_small_llm_model(scene: str = "general") -> str:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        return _pick_env(
            "AIOPS_SMALL_LLM_MODEL",
            "SMALL_LLM_MODEL",
            "AIOPS_LOCAL_LLM_MODEL",
            "LOCAL_LLM_MODEL",
            default=_DEFAULT_SMALL_MODEL,
        ) or _DEFAULT_SMALL_MODEL
    return _pick_env(
        f"AIOPS_{scene_key}_SMALL_LLM_MODEL",
        f"SMALL_{scene_key}_LLM_MODEL",
        f"AIOPS_{scene_key}_LOCAL_LLM_MODEL",
        f"LOCAL_{scene_key}_LLM_MODEL",
        "AIOPS_SMALL_LLM_MODEL",
        "SMALL_LLM_MODEL",
        "AIOPS_LOCAL_LLM_MODEL",
        "LOCAL_LLM_MODEL",
        default=_DEFAULT_SMALL_MODEL,
    ) or _DEFAULT_SMALL_MODEL


def _resolve_small_llm_base_url(scene: str = "general") -> str:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        return _pick_env(
            "AIOPS_SMALL_LLM_BASE_URL",
            "SMALL_LLM_BASE_URL",
            "AIOPS_LOCAL_LLM_BASE_URL",
            "LOCAL_LLM_BASE_URL",
            default=_DEFAULT_SMALL_BASE_URL,
        ) or _DEFAULT_SMALL_BASE_URL
    return _pick_env(
        f"AIOPS_{scene_key}_SMALL_LLM_BASE_URL",
        f"SMALL_{scene_key}_LLM_BASE_URL",
        f"AIOPS_{scene_key}_LOCAL_LLM_BASE_URL",
        f"LOCAL_{scene_key}_LLM_BASE_URL",
        "AIOPS_SMALL_LLM_BASE_URL",
        "SMALL_LLM_BASE_URL",
        "AIOPS_LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_BASE_URL",
        default=_DEFAULT_SMALL_BASE_URL,
    ) or _DEFAULT_SMALL_BASE_URL


def _resolve_small_llm_think(scene: str = "general") -> bool:
    scene_key = _scene_key(scene)
    if scene_key == "GENERAL":
        raw = _pick_env("AIOPS_SMALL_LLM_THINK", "SMALL_LLM_THINK", default="false")
    else:
        raw = _pick_env(
            f"AIOPS_{scene_key}_SMALL_LLM_THINK",
            f"SMALL_{scene_key}_LLM_THINK",
            "AIOPS_SMALL_LLM_THINK",
            "SMALL_LLM_THINK",
            default="false",
        )
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_small_llm_max_tokens(scene: str = "general") -> int:
    scene_key = _scene_key(scene)
    default_tokens = _default_small_llm_max_tokens(scene)
    if scene_key == "GENERAL":
        raw = _pick_env("AIOPS_SMALL_LLM_MAX_TOKENS", "SMALL_LLM_MAX_TOKENS", default=str(default_tokens))
    else:
        raw = _pick_env(
            f"AIOPS_{scene_key}_SMALL_LLM_MAX_TOKENS",
            f"SMALL_{scene_key}_LLM_MAX_TOKENS",
            "AIOPS_SMALL_LLM_MAX_TOKENS",
            "SMALL_LLM_MAX_TOKENS",
            default=str(default_tokens),
        )
    max_tokens = _safe_int(raw)
    if max_tokens is None:
        return default_tokens
    return max(max_tokens, 1)


def _should_force_json_output(scene: str = "general") -> bool:
    return _scene_key(scene) in {"INTENT", "SENSITIVE"}


def _build_small_llm_client(base_url: str) -> Any | None:
    global _SMALL_LLM_INIT_DONE, _SMALL_LLM_CLIENT, _SMALL_LLM_CLIENT_CACHE
    if not _SMALL_LLM_INIT_DONE and _SMALL_LLM_CLIENT is None and _SMALL_LLM_CLIENT_CACHE:
        _SMALL_LLM_CLIENT_CACHE = {}
    if ollama is None:
        return None

    if base_url in _SMALL_LLM_CLIENT_CACHE:
        return _SMALL_LLM_CLIENT_CACHE[base_url]

    try:
        client = ollama.Client(host=base_url)
    except Exception:  # noqa: BLE001
        return None

    _SMALL_LLM_CLIENT_CACHE[base_url] = client
    if not _SMALL_LLM_INIT_DONE:
        _SMALL_LLM_INIT_DONE = True
        _SMALL_LLM_CLIENT = client
    return client


def _invoke_small_llm(system_prompt: str, user_prompt: str, *, scene: str = "general") -> str:
    model = _resolve_small_llm_model(scene)
    base_url = _resolve_small_llm_base_url(scene)
    think_enabled = _resolve_small_llm_think(scene)
    max_tokens = _resolve_small_llm_max_tokens(scene)
    force_json_output = _should_force_json_output(scene)
    llm = _build_small_llm_client(base_url=base_url)
    if llm is None:
        _SMALL_LLM_LOG.warning(
            "small_llm.invoke.skip scene=%s model=%s base_url=%s reason=no_client",
            scene,
            model,
            base_url,
        )
        return ""

    _SMALL_LLM_LOG.info(
        "small_llm.invoke.start scene=%s model=%s base_url=%s think=%s max_tokens=%d force_json=%s user_chars=%d system_chars=%d",
        scene,
        model,
        base_url,
        str(think_enabled).lower(),
        max_tokens,
        str(force_json_output).lower(),
        len(str(user_prompt or "")),
        len(str(system_prompt or "")),
    )
    started = time.perf_counter()
    try:
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "think": think_enabled,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if force_json_output:
            chat_kwargs["format"] = "json"
        result = llm.chat(**chat_kwargs)
        payload = _coerce_dict(result)
        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else ""
        out = _coerce_text(content).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        input_tokens, output_tokens, total_tokens = _extract_token_usage(result)
        _SMALL_LLM_LOG.info(
            "small_llm.invoke.end scene=%s model=%s base_url=%s out_chars=%d duration_ms=%.2f input_tokens=%s output_tokens=%s total_tokens=%s",
            scene,
            model,
            base_url,
            len(out),
            duration_ms,
            str(input_tokens) if input_tokens is not None else "n/a",
            str(output_tokens) if output_tokens is not None else "n/a",
            str(total_tokens) if total_tokens is not None else "n/a",
        )
        return out
    except Exception:  # noqa: BLE001
        duration_ms = (time.perf_counter() - started) * 1000
        _SMALL_LLM_LOG.exception(
            "small_llm.invoke.error scene=%s model=%s base_url=%s duration_ms=%.2f",
            scene,
            model,
            base_url,
            duration_ms,
        )
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


def chat_with_small_llm(question: str, system_prompt: str = "") -> str:
    question_text = str(question or "").strip()
    if not question_text:
        return ""
    return _invoke_small_llm(system_prompt=system_prompt, user_prompt=question_text, scene="general")


def recognize_intent_with_small_llm(question: str, intent_history_prompt: str | None = None) -> dict[str, Any]:
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

    text = _invoke_small_llm(system_prompt, user_prompt, scene="intent_recognition")
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


def check_sensitive_operation_with_small_llm(question: str) -> dict[str, Any]:
    question_text = str(question or "").strip()
    if not question_text:
        return {"passed": False, "reason": "empty question"}

    system_prompt = load_prompt("sensitive_operation_system_prompt.txt", default="")
    user_prompt = render_prompt("sensitive_operation_user_prompt.txt", question=question_text)
    if not user_prompt:
        return {"passed": False, "reason": "sensitive prompt missing"}

    text = _invoke_small_llm(system_prompt, user_prompt, scene="sensitive_check")
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


def get_small_llm_runtime_config() -> dict[str, str]:
    return {
        "model": _resolve_small_llm_model("general"),
        "base_url": _resolve_small_llm_base_url("general"),
    }
