"""Qdrant vector store wrapper.

Reads local Qdrant configuration from environment and reuses
`embedding.text_embedding` for vectorization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from embedding.embedding import text_embedding

try:  # Optional runtime dependency.
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, PointStruct, VectorParams
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    Distance = None  # type: ignore[assignment]
    PointStruct = None  # type: ignore[assignment]
    VectorParams = None  # type: ignore[assignment]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class QdrantConfig:
    url: str
    collection_name: str
    vector_dim: int
    timeout_sec: float

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        url = str(os.getenv("QDRANT_URL", "http://127.0.0.1:6333")).strip() or "http://127.0.0.1:6333"
        collection_name = (
            str(os.getenv("QDRANT_COLLECTION_NAME", "ai_ops_rag_demo")).strip() or "ai_ops_rag_demo"
        )
        vector_dim = max(1, _to_int(os.getenv("QDRANT_VECTOR_DIM", "512"), 512))
        timeout_sec = float(os.getenv("QDRANT_TIMEOUT_SEC", "3"))
        return cls(
            url=url,
            collection_name=collection_name,
            vector_dim=vector_dim,
            timeout_sec=max(0.1, timeout_sec),
        )


class QdrantStore:
    """Simple upsert/search wrapper for local Qdrant."""

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config = config or QdrantConfig.from_env()
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if QdrantClient is None or Distance is None or VectorParams is None:
            raise RuntimeError("qdrant_client is required. Install dependency: qdrant-client")
        self._client = QdrantClient(url=self.config.url, timeout=self.config.timeout_sec)
        return self._client

    def ensure_collection(self) -> None:
        client = self._ensure_client()
        collections = client.get_collections()
        names = {row.name for row in list(collections.collections or [])}
        if self.config.collection_name in names:
            return
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=self.config.vector_dim, distance=Distance.COSINE),
        )

    def upsert_text(
        self,
        *,
        text: str,
        point_id: str | int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str | int:
        content = str(text or "").strip()
        if not content:
            raise ValueError("text must not be empty")
        self.ensure_collection()
        client = self._ensure_client()
        pid: str | int = point_id if point_id is not None else uuid4().hex
        vector = text_embedding(content, dim=self.config.vector_dim)
        point_payload = {"text": content, **dict(payload or {})}
        client.upsert(
            collection_name=self.config.collection_name,
            points=[PointStruct(id=pid, vector=vector, payload=point_payload)],
        )
        return pid

    def upsert_texts(self, items: list[dict[str, Any]]) -> list[str | int]:
        self.ensure_collection()
        client = self._ensure_client()
        rows: list[Any] = []
        ids: list[str | int] = []
        for item in list(items or []):
            content = str(item.get("text") or "").strip()
            if not content:
                continue
            pid: str | int = item.get("id") if item.get("id") is not None else uuid4().hex
            vector = text_embedding(content, dim=self.config.vector_dim)
            payload = dict(item.get("payload") or {})
            rows.append(PointStruct(id=pid, vector=vector, payload={"text": content, **payload}))
            ids.append(pid)
        if rows:
            client.upsert(collection_name=self.config.collection_name, points=rows)
        return ids

    def search(
        self,
        *,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        self.ensure_collection()
        client = self._ensure_client()
        query_vector = text_embedding(query_text, dim=self.config.vector_dim)
        points = client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            limit=max(1, int(limit)),
            score_threshold=score_threshold,
        )
        rows: list[dict[str, Any]] = []
        for row in list(points or []):
            rows.append(
                {
                    "id": getattr(row, "id", None),
                    "score": float(getattr(row, "score", 0.0) or 0.0),
                    "payload": dict(getattr(row, "payload", {}) or {}),
                }
            )
        return rows
