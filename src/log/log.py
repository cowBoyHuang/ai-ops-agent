"""External log API client."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


def _read_env_file() -> dict[str, str]:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _env(key: str, fallback: str = "", file_values: dict[str, str] | None = None) -> str:
    if key in os.environ:
        return str(os.environ[key])
    if file_values and key in file_values:
        return str(file_values[key])
    return fallback


def _canonical_json_bytes(obj: Any) -> bytes:
    body = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return body.encode("utf-8")


def _hmac_sha256_base64(secret: str, message: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), message, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


@dataclass
class LogApiConfig:
    base_url: str
    endpoint_path: str
    request_source: str
    app_key: str
    app_secret: str
    sign_alg: str
    sign_mode: str
    token: str
    timeout_sec: int

    @classmethod
    def from_env(cls) -> "LogApiConfig":
        file_values = _read_env_file()
        timeout_raw = _env("LOG_API_TIMEOUT_SEC", "6", file_values)
        try:
            timeout_val = max(1, int(timeout_raw))
        except ValueError:
            timeout_val = 6
        return cls(
            base_url=_env("LOG_API_BASE_URL", _env("LOG_API_URL", "", file_values), file_values),
            endpoint_path=_env("LOG_API_ENDPOINT_PATH", "/api/maintenance/pullLogByCondition", file_values),
            request_source=_env("LOG_API_REQUEST_SOURCE", "", file_values),
            app_key=_env("LOG_API_APP_KEY", "", file_values),
            app_secret=_env("LOG_API_APP_SECRET", "", file_values),
            sign_alg=_env("LOG_API_SIGN_ALG", "HMAC-SHA256", file_values),
            sign_mode=_env("LOG_API_SIGN_MODE", "REQUEST", file_values),
            token=_env("LOG_API_TOKEN", "", file_values),
            timeout_sec=timeout_val,
        )


@dataclass
class EsResult:
    score: float
    content: str


def _build_headers(config: LogApiConfig, payload_bytes: bytes) -> dict[str, str]:
    ts_ms = str(int(time.time() * 1000))
    nonce = uuid4().hex
    sign_message = b"|".join(
        [
            config.app_key.encode("utf-8"),
            config.request_source.encode("utf-8"),
            ts_ms.encode("utf-8"),
            nonce.encode("utf-8"),
            payload_bytes,
        ]
    )
    signature = _hmac_sha256_base64(config.app_secret, sign_message) if config.app_secret else ""
    headers = {
        "Content-Type": "application/json",
        "requestSource": config.request_source,
        "appKey": config.app_key,
        "signAlg": config.sign_alg,
        "signMode": config.sign_mode,
        "timestamp": ts_ms,
        "nonce": nonce,
        "sign": signature,
    }
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    return headers


def build_pull_log_request(
    *,
    app_code: str,
    keyword: str = "",
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    window_minutes: int = 15,
    page_no: int = 1,
    page_size: int = 200,
    log_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    构建 pullLogByCondition 请求体。

    说明：
    - 未传 start/end 时，默认使用“当前时间往前 window_minutes 分钟”。
    - 时间统一转 ISO8601 字符串，便于接口直接消费。
    """
    now = dt.datetime.now(dt.timezone.utc)
    final_end = end_time or now
    final_start = start_time or (final_end - dt.timedelta(minutes=max(1, window_minutes)))
    if not log_name.strip():
        raise ValueError("log_name is required")
    payload: dict[str, Any] = {
        "appCode": app_code,
        "keyword": keyword,
        "startTime": final_start.isoformat(),
        "endTime": final_end.isoformat(),
        "pageNo": max(1, int(page_no)),
        "pageSize": max(1, int(page_size)),
        "logName": log_name,
    }
    if extra:
        payload.update(extra)
    return payload


def pull_log_by_condition(condition: dict[str, Any], config: LogApiConfig | None = None) -> dict[str, Any]:
    cfg = config or LogApiConfig.from_env()
    if not cfg.base_url:
        raise RuntimeError("LOG_API_BASE_URL / LOG_API_URL is empty")

    url = f"{cfg.base_url.rstrip('/')}/{cfg.endpoint_path.lstrip('/')}"
    payload_bytes = _canonical_json_bytes(condition)
    headers = _build_headers(cfg, payload_bytes)
    request = urllib.request.Request(url=url, data=payload_bytes, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)

    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"log api http error {err.code}: {body}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"log api url error: {err}") from err

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def build_external_log_request(
    *,
    app_core: str,
    content: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    logname: str,
    type: str,
) -> dict[str, Any]:
    """构建外部日志查询请求（必填参数版本）。"""
    if not app_core.strip():
        raise ValueError("app_core is required")
    if not content.strip():
        raise ValueError("content is required")
    if not logname.strip():
        raise ValueError("logname is required")
    if not type.strip():
        raise ValueError("type is required")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    payload = build_pull_log_request(
        app_code=app_core,
        keyword=content,
        start_time=start_time,
        end_time=end_time,
        page_no=1,
        page_size=200,
        log_name=logname,
    )
    # 兼容潜在不同字段命名
    payload["appCore"] = app_core
    payload["content"] = content
    payload["type"] = type
    return payload


def adapt_raw_item_to_es_result(raw_item: dict[str, Any]) -> EsResult:
    """将单条原始对象适配为 EsResult。"""
    source = raw_item.get("_source") if isinstance(raw_item, dict) else None
    score_val = (
        raw_item.get("_score")
        if isinstance(raw_item, dict)
        else None
    )
    if score_val is None and isinstance(raw_item, dict):
        score_val = raw_item.get("score", 0)
    try:
        score = float(score_val or 0)
    except (TypeError, ValueError):
        score = 0.0

    content = ""
    if isinstance(raw_item, dict):
        content = str(raw_item.get("content") or raw_item.get("message") or raw_item.get("logContent") or "")
    if not content and isinstance(source, dict):
        content = str(source.get("content") or source.get("message") or "")
    if not content:
        content = str(raw_item)
    return EsResult(score=score, content=content)


def adapt_raw_response_to_es_results(raw: dict[str, Any]) -> list[EsResult]:
    """将接口原始返回适配为 EsResult 列表。"""
    candidates: list[dict[str, Any]] = []
    if isinstance(raw.get("hits"), dict) and isinstance(raw["hits"].get("hits"), list):
        candidates = [item for item in raw["hits"]["hits"] if isinstance(item, dict)]
    elif isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("list"), list):
        candidates = [item for item in raw["data"]["list"] if isinstance(item, dict)]
    elif isinstance(raw.get("list"), list):
        candidates = [item for item in raw["list"] if isinstance(item, dict)]
    elif isinstance(raw.get("rows"), list):
        candidates = [item for item in raw["rows"] if isinstance(item, dict)]
    return [adapt_raw_item_to_es_result(item) for item in candidates]


def query_external_logs(
    *,
    app_core: str,
    content: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    logname: str,
    type: str,
    config: LogApiConfig | None = None,
) -> list[EsResult]:
    """
    外部日志查询包装方法：
    必填参数 -> 构建请求 -> 调接口 -> 适配结果。
    """
    request_body = build_external_log_request(
        app_core=app_core,
        content=content,
        start_time=start_time,
        end_time=end_time,
        logname=logname,
        type=type,
    )
    raw = pull_log_by_condition(request_body, config=config)
    return adapt_raw_response_to_es_results(raw)

