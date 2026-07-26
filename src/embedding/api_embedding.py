"""HTTP client for Qunar online embedding models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Sequence

import httpx


# Keep this aligned with src/llm/llm.py. The embeddings endpoint is appended
# below so callers may pass either the /v1 base URL or the service root.
_DEFAULT_BASE_URL = "http://llm.api.corp.qunar.com/v1"
_DEFAULT_MODEL = "azure/text-embedding-3-small"
_DEFAULT_DIMENSIONS = 512
_DEFAULT_TIMEOUT_SEC = 30.0
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 10
_DEFAULT_MAX_CONNECTIONS = 20
_DEFAULT_KEEPALIVE_EXPIRY_SEC = 120.0
_RECENT_DURATION_WINDOW = 5
_LOGGER = logging.getLogger("aiops.embedding")
_SHARED_CLIENT: Any | None = None


def _pick_env(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default


def _embedding_url(base_url: str) -> str:
    normalized = str(base_url or _DEFAULT_BASE_URL).strip().rstrip("/")
    if not normalized:
        normalized = _DEFAULT_BASE_URL
    if normalized.endswith("/v1"):
        return f"{normalized}/embeddings"
    return f"{normalized}/v1/embeddings"


def _optional_int_env(name: str) -> int | None:
    value = str(os.getenv(name, "")).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class ApiEmbeddingConfig:
    base_url: str = _DEFAULT_BASE_URL
    model: str = _DEFAULT_MODEL
    api_key: str = ""
    dimensions: int | None = _DEFAULT_DIMENSIONS
    encoding_format: str = "float"
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC
    max_keepalive_connections: int = _DEFAULT_MAX_KEEPALIVE_CONNECTIONS
    max_connections: int = _DEFAULT_MAX_CONNECTIONS
    keepalive_expiry_sec: float = _DEFAULT_KEEPALIVE_EXPIRY_SEC

    @classmethod
    def from_env(cls) -> "ApiEmbeddingConfig":
        timeout_raw = _pick_env("AIOPS_EMBEDDING_TIMEOUT_SEC", default=str(_DEFAULT_TIMEOUT_SEC))
        try:
            timeout_sec = float(timeout_raw)
        except ValueError:
            timeout_sec = _DEFAULT_TIMEOUT_SEC
        keepalive_raw = _pick_env(
            "AIOPS_EMBEDDING_KEEPALIVE_EXPIRY_SEC",
            default=str(_DEFAULT_KEEPALIVE_EXPIRY_SEC),
        )
        try:
            keepalive_expiry_sec = float(keepalive_raw)
        except ValueError:
            keepalive_expiry_sec = _DEFAULT_KEEPALIVE_EXPIRY_SEC
        max_keepalive_connections = max(
            1,
            _optional_int_env("AIOPS_EMBEDDING_MAX_KEEPALIVE_CONNECTIONS")
            or _DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        )
        max_connections = max(
            max_keepalive_connections,
            _optional_int_env("AIOPS_EMBEDDING_MAX_CONNECTIONS") or _DEFAULT_MAX_CONNECTIONS,
        )
        return cls(
            base_url=_pick_env(
                "AIOPS_EMBEDDING_BASE_URL",
                "AIOPS_LLM_BASE_URL",
                "LLM_BASE_URL",
                default=_DEFAULT_BASE_URL,
            ),
            model=_pick_env("AIOPS_EMBEDDING_MODEL", default=_DEFAULT_MODEL),
            api_key=_pick_env("OPENAI_API_KEY", "LLM_API_KEY", default=""),
            dimensions=_optional_int_env("AIOPS_EMBEDDING_DIMENSIONS") or _DEFAULT_DIMENSIONS,
            encoding_format=_pick_env("AIOPS_EMBEDDING_ENCODING_FORMAT", default="float"),
            timeout_sec=max(0.1, timeout_sec),
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry_sec=max(0.1, keepalive_expiry_sec),
        )


@dataclass(frozen=True)
class ApiEmbeddingResult:
    model: str
    embeddings: list[list[float]]
    usage: dict[str, Any]
    raw: dict[str, Any]

    @property
    def input_tokens(self) -> int | None:
        value = self.usage.get("prompt_tokens", self.usage.get("input_tokens"))
        return int(value) if isinstance(value, int | float) else None

    @property
    def total_tokens(self) -> int | None:
        value = self.usage.get("total_tokens")
        return int(value) if isinstance(value, int | float) else None

    @property
    def dimensions(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


class ApiEmbeddingClient:
    def __init__(
        self,
        config: ApiEmbeddingConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or ApiEmbeddingConfig.from_env()
        self._transport = transport
        self._client: httpx.Client | None = None
        self._recent_duration_ms: deque[float] = deque(maxlen=_RECENT_DURATION_WINDOW)

    def _get_client(self) -> httpx.Client:
        if self._client is not None and not self._client.is_closed:
            return self._client
        self._client = httpx.Client(
            timeout=self.config.timeout_sec,
            transport=self._transport,
            limits=httpx.Limits(
                max_keepalive_connections=self.config.max_keepalive_connections,
                max_connections=self.config.max_connections,
                keepalive_expiry=self.config.keepalive_expiry_sec,
            ),
        )
        return self._client

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _request_embeddings(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY/LLM_API_KEY is required for online embedding")
        started = time.perf_counter()
        response = self._get_client().post(_embedding_url(self.config.base_url), json=payload, headers=self._headers())
        response.raise_for_status()
        duration_ms = (time.perf_counter() - started) * 1000
        body = response.json()
        return dict(body), duration_ms

    def _record_duration(self, duration_ms: float, *, input_count: int, operation: str) -> None:
        self._recent_duration_ms.append(float(duration_ms))
        avg_ms = sum(self._recent_duration_ms) / max(1, len(self._recent_duration_ms))
        _LOGGER.info(
            "embedding.%s.end model=%s input_count=%d duration_ms=%.2f recent_avg_ms=%.2f recent_count=%d",
            operation,
            self.config.model,
            input_count,
            duration_ms,
            avg_ms,
            len(self._recent_duration_ms),
        )

    def _build_payload(self, normalized_input: str | list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": normalized_input,
            "model": self.config.model,
        }
        if self.config.dimensions is not None:
            payload["dimensions"] = self.config.dimensions
        if self.config.encoding_format:
            payload["encoding_format"] = self.config.encoding_format
        return payload

    def embed(self, inputs: str | Sequence[str], *, user: str | None = None) -> ApiEmbeddingResult:
        normalized_input: str | list[str]
        if isinstance(inputs, str):
            normalized_input = inputs
        else:
            normalized_input = [str(item) for item in inputs]
        if isinstance(normalized_input, str):
            if not normalized_input.strip():
                raise ValueError("input must not be empty")
        elif not any(item.strip() for item in normalized_input):
            raise ValueError("input must not be empty")
        payload = self._build_payload(normalized_input)
        if user:
            payload["user"] = user

        body, duration_ms = self._request_embeddings(payload)
        input_count = 1 if isinstance(normalized_input, str) else len(normalized_input)
        self._record_duration(duration_ms, input_count=input_count, operation="batch")

        data = list(body.get("data") or [])
        embeddings: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                continue
            embeddings.append([float(value) for value in embedding])
        if not embeddings:
            raise RuntimeError("embedding response contains no vectors")

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return ApiEmbeddingResult(
            model=str(body.get("model") or self.config.model),
            embeddings=embeddings,
            usage=dict(usage or {}),
            raw=dict(body),
        )

    def warm_up_empty_batch(self) -> float:
        payload = self._build_payload([])
        _, duration_ms = self._request_embeddings(payload)
        self._record_duration(duration_ms, input_count=0, operation="warmup")
        return duration_ms


def get_shared_embedding_client(config: ApiEmbeddingConfig | None = None) -> ApiEmbeddingClient:
    global _SHARED_CLIENT
    if config is not None:
        if _SHARED_CLIENT is not None:
            _SHARED_CLIENT.close()
        _SHARED_CLIENT = ApiEmbeddingClient(config=config)
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = ApiEmbeddingClient()
    return _SHARED_CLIENT


def warm_up_shared_embedding_client() -> float:
    return get_shared_embedding_client().warm_up_empty_batch()


def close_shared_embedding_client() -> None:
    if _SHARED_CLIENT is not None:
        _SHARED_CLIENT.close()


def embed_texts(inputs: str | Sequence[str], config: ApiEmbeddingConfig | None = None) -> ApiEmbeddingResult:
    return get_shared_embedding_client(config=config).embed(inputs)
