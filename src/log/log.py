"""External log API client."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

# JinkelaClient.java 默认配置（用于本地未配置环境变量时兜底）
_DEFAULT_ES_SCHEME = "http"
_DEFAULT_ES_PORT = 80
_DEFAULT_ES_HOST = "jinkelalinkflight.corp.qunar.com"
_DEFAULT_ES_USERNAME = "system_f_pangu-ordertool"
_DEFAULT_ES_PASSWORD = "iVa!ChHlx%7dJN8"
_DEFAULT_ES_API_KEY = ""
_LOGGER = logging.getLogger(__name__)
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


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


def _to_pangu_datetime_text(value: dt.datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(_LOCAL_TZ).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_log_name_for_es(raw: str) -> str:
    text = raw.strip()
    if text.endswith("*"):
        text = text[:-1].strip()
    dot_index = text.rfind(".")
    if dot_index > 0:
        text = text[:dot_index].strip()
    return text or raw.strip()


def _build_match_rule(*, match_type: str, field_name: str, pattern: str) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "type": match_type,
        "fieldName": field_name,
        "pattern": pattern,
    }
    # Java 侧 CONTAINS/CONTAINS_ANY 会兼容读取 terms，这里一起带上。
    if match_type in {"CONTAINS", "CONTAINS_ANY", "CONTAINS_ALL", "NOT_CONTAINS"}:
        rule["terms"] = [pattern]
    return rule


class QueryType(StrEnum):
    MATCH_PHRASE = "match_phrase"
    MATCH = "match"


def _parse_query_type(raw: str) -> QueryType:
    value = (raw or "").strip().lower()
    if value == QueryType.MATCH_PHRASE.value:
        return QueryType.MATCH_PHRASE
    if value == QueryType.MATCH.value:
        return QueryType.MATCH
    raise ValueError("type must be one of: match_phrase, match")


def _coerce_time_text(value: dt.datetime | str, field_name: str) -> str:
    if isinstance(value, dt.datetime):
        return _to_pangu_datetime_text(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    # 输入如果是 ISO8601，统一转 yyyy-MM-dd HH:mm:ss
    iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(iso_candidate)
        return _to_pangu_datetime_text(parsed)
    except ValueError:
        return text


def _normalize_base_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return f"http://{text}"


@dataclass
class LogApiConfig:
    base_url: str
    endpoint_path: str
    token: str
    es_api_key: str
    es_username: str
    es_password: str
    timeout_sec: int

    @classmethod
    def from_env(cls) -> "LogApiConfig":
        file_values = _read_env_file()
        timeout_raw = _env("LOG_API_TIMEOUT_SEC", "6", file_values)
        try:
            timeout_val = max(1, int(timeout_raw))
        except ValueError:
            timeout_val = 6
        # 按用户要求：以下 4 个字段只使用文件内配置，不读取环境变量：
        # - ES_API_BASE_URL
        # - ES_API_KEY
        # - ES_USERNAME
        # - ES_PASSWORD
        base_url = _normalize_base_url(f"{_DEFAULT_ES_SCHEME}://{_DEFAULT_ES_HOST}:{_DEFAULT_ES_PORT}")
        endpoint_path = _env("ES_API_ENDPOINT_PATH", "_search", file_values)

        return cls(
            base_url=base_url,
            endpoint_path=endpoint_path,
            token=_env("LOG_API_TOKEN", "", file_values),
            es_api_key=_DEFAULT_ES_API_KEY,
            es_username=_DEFAULT_ES_USERNAME,
            es_password=_DEFAULT_ES_PASSWORD,
            timeout_sec=timeout_val,
        )


@dataclass
class EsResult:
    score: float
    content: str


def _build_headers(config: LogApiConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.es_api_key:
        headers["Authorization"] = f"ApiKey {config.es_api_key}"
    elif config.es_username and config.es_password:
        basic = base64.b64encode(f"{config.es_username}:{config.es_password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {basic}"
    elif config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    return headers


def _get_index_name(app_code: str) -> str:
    return f"log_{app_code}-*"


def _build_content_clause(content: str, query_type: QueryType) -> dict[str, Any]:
    if query_type == QueryType.MATCH:
        # 针对“关键词+大JSON”场景：短语优先 + BM25 兜底。
        return {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "content": {
                                "query": content,
                                "boost": 2.0,
                            },
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": content,
                                "analyzer": "ik_max_word",
                                "minimum_should_match": "20%",
                                "boost": 1.0,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    else:  # QueryType.MATCH_PHRASE
        return {query_type.value: {"content": content}}

def build_pull_log_request(
    *,
    app_code: str,
    keyword: str = "",
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    window_minutes: int = 15,
    page_no: int = 1,  # compatibility placeholder
    page_size: int = 200,
    log_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del page_no  # 兼容旧参数
    del window_minutes  # 兼容旧参数
    if not start_time or not end_time:
        raise ValueError("start_time and end_time are required")
    return build_es_pull_log_request(
        app_code=app_code,
        logname=log_name,
        begin_time=start_time,
        end_time=end_time,
        content=keyword,
        type=QueryType.MATCH.value,
        max_lines=page_size,
        extra=extra,
    )


def build_es_pull_log_request(
    *,
    app_code: str,
    logname: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    content: str,
    type: str,
    max_lines: int = 200,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    ES 请求封装方法（方法1内部调用）。

    必要参数：
    - app_code, logname, begin_time(beginTime), end_time(endTime), content, type(match_phrase|match)
    """
    if not app_code.strip():
        raise ValueError("app_code is required")
    if not logname.strip():
        raise ValueError("logname is required")
    if not content.strip():
        raise ValueError("content is required")
    query_type = _parse_query_type(type)

    begin_text = _coerce_time_text(begin_time, "begin_time")
    end_text = _coerce_time_text(end_time, "end_time")
    try:
        begin_dt = dt.datetime.fromisoformat(begin_text)
        end_dt = dt.datetime.fromisoformat(end_text)
        if end_dt <= begin_dt:
            raise ValueError("end_time must be greater than begin_time")
    except ValueError:
        # 时间格式允许由 ES 按字符串处理，不强制中断
        pass
    normalized_log_name = _normalize_log_name_for_es(logname)

    body: dict[str, Any] = {
        "size": max(1, int(max_lines)),
        "sort": [{"@timestamp": {"order": "asc"}}],
        "_source": ["content"],
        "track_scores": True,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": begin_text,
                                "lte": end_text,
                                "time_zone": "+08:00",
                                "format": "yyyy-MM-dd HH:mm:ss||strict_date_optional_time",
                            }
                        }
                    },
                    _build_content_clause(content.strip(), query_type),
                    {"term": {"log_name": normalized_log_name}},
                ]
            }
        },
    }
    request_payload: dict[str, Any] = {
        "index": _get_index_name(app_code.strip()),
        "body": body,
        "appCode": app_code.strip(),
        "beginTime": begin_text,
        "endTime": end_text,
        "logname": logname.strip(),
        "content": content.strip(),
        "type": query_type.value,
    }
    if extra:
        request_payload.update(extra)
    return request_payload


def build_legacy_pull_log_request_for_api(
    *,
    app_code: str,
    keyword: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    log_name: str,
    max_lines: int = 200,
) -> dict[str, Any]:
    """
    兼容旧结构：如需回退老接口格式可复用此方法（当前主链路不使用）。
    """
    begin_text = _coerce_time_text(begin_time, "begin_time")
    end_text = _coerce_time_text(end_time, "end_time")
    match_rules: list[dict[str, Any]] = [
        _build_match_rule(match_type="CONTAINS", field_name="content", pattern=keyword.strip()),
        _build_match_rule(match_type="CONTAINS", field_name="log_name", pattern=_normalize_log_name_for_es(log_name)),
    ]
    return {
        "appCode": app_code.strip(),
        "beginTime": begin_text,
        "endTime": end_text,
        "maxLen": max(1, int(max_lines)),
        "include": "content",
        "condition": {
            "logicOpr": "AND",
            "childLogicOpr": "AND",
            "matchRules": match_rules,
            "childConditions": [],
        },
    }


def pull_log_by_condition(condition: dict[str, Any], config: LogApiConfig | None = None) -> dict[str, Any]:

    cfg = config or LogApiConfig.from_env()
    if not cfg.base_url:
        raise RuntimeError("ES_API_BASE_URL / ES_BASE_URL is empty")

    index = str(condition.get("index") or _get_index_name(str(condition.get("appCode") or ""))).strip()
    if not index:
        raise RuntimeError("es index is empty")

    body = condition.get("body") if isinstance(condition.get("body"), dict) else condition
    url = f"{cfg.base_url.rstrip('/')}/{index}/{cfg.endpoint_path.lstrip('/')}"
    payload_bytes = _canonical_json_bytes(body)
    headers = _build_headers(cfg)
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
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}

    if isinstance(parsed, dict) and parsed.get("success") is False:
        err_msg = str(parsed.get("errMsg") or parsed.get("message") or "unknown error")
        raise RuntimeError(f"log api business error: {err_msg}")
    return parsed


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
    payload = build_es_pull_log_request(
        app_code=app_core,
        logname=logname,
        begin_time=start_time,
        end_time=end_time,
        content=content,
        type=type,
        max_lines=200,
    )
    payload["appCore"] = app_core  # 兼容旧调用方字段。
    return payload


def adapt_raw_item_to_es_result(raw_item: Any) -> EsResult:
    """将单条原始对象适配为 EsResult。"""
    if isinstance(raw_item, str):
        return EsResult(score=0.0, content=raw_item)

    source = raw_item.get("_source") if isinstance(raw_item, dict) else None
    score_val = raw_item.get("_score") if isinstance(raw_item, dict) else None
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


def _extract_candidates(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []

    # Result<List<String>> / Result<List<Map>>
    if isinstance(raw.get("data"), list):
        return raw["data"]
    if isinstance(raw.get("data"), dict):
        nested = _extract_candidates(raw["data"])
        if nested:
            return nested

    # standard es response
    hits = raw.get("hits")
    if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
        return hits["hits"]

    # common list envelope
    for key in ("list", "rows"):
        value = raw.get(key)
        if isinstance(value, list):
            return value

    return []


def adapt_raw_response_to_es_results(raw: dict[str, Any]) -> list[EsResult]:
    """
    原始结果处理方法：
    把日志原始结果 hit 命中项适配为 EsResult 列表。
    """
    candidates = _extract_candidates(raw)
    return [adapt_raw_item_to_es_result(item) for item in candidates]


def query_external_logs(
    *,
    app_code: str | None = None,
    logname: str = "",
    begin_time: dt.datetime | str | None = None,
    end_time: dt.datetime | str | None = None,
    content: str = "",
    type: str = "",
    config: LogApiConfig | None = None,
    # legacy aliases
    app_core: str | None = None,
    start_time: dt.datetime | str | None = None,
    beginTime: dt.datetime | str | None = None,
    endTime: dt.datetime | str | None = None,
) -> list[EsResult]:
    """
    外部日志查询包装方法：
    必填参数 -> 构建请求 -> 调接口 -> 适配结果。
    """
    final_app_code = (app_code or app_core or "").strip()
    if not final_app_code:
        raise ValueError("app_code is required")
    final_begin = begin_time if begin_time is not None else beginTime if beginTime is not None else start_time
    final_end = end_time if end_time is not None else endTime
    if final_begin is None or final_end is None:
        raise ValueError("begin_time/end_time are required")

    _LOGGER.info("log.query_external_logs.start app_code=%s logname=%s", final_app_code, str(logname or ""))
    rows = search_logs(
        app_code=final_app_code,
        logname=logname,
        begin_time=final_begin,
        end_time=final_end,
        content=content,
        type=type,
        config=config,
    )
    _LOGGER.info("log.query_external_logs.end app_code=%s hit_count=%d", final_app_code, len(rows))
    return rows


def search_logs(
    *,
    app_code: str,
    logname: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    content: str,
    type: str,
    config: LogApiConfig | None = None,
) -> list[EsResult]:
    """
    对外日志查询方法（原生 ES）：
    1. 封装 ES 请求
    2. 调用 ES 外部接口
    3. 处理原始 hits 结果并返回 List[EsResult]
    """
    request_body = build_es_pull_log_request(
        app_code=app_code,
        logname=logname,
        begin_time=begin_time,
        end_time=end_time,
        content=content,
        type=type,
    )
    raw = pull_log_by_condition(request_body, config=config)
    return adapt_raw_response_to_es_results(raw)
