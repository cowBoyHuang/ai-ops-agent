"""Qdrant vector store wrapper.

Reads local Qdrant configuration from environment and reuses
`embedding.text_embedding` for vectorization.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from embedding.embedding import text_embedding
from rerank.reranker import rerank_documents

try:  # Optional runtime dependency.
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        Fusion,
        FusionQuery,
        Modifier,
        PointStruct,
        Prefetch,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    Distance = None  # type: ignore[assignment]
    Fusion = None  # type: ignore[assignment]
    FusionQuery = None  # type: ignore[assignment]
    Modifier = None  # type: ignore[assignment]
    PointStruct = None  # type: ignore[assignment]
    Prefetch = None  # type: ignore[assignment]
    SparseVector = None  # type: ignore[assignment]
    SparseVectorParams = None  # type: ignore[assignment]
    VectorParams = None  # type: ignore[assignment]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _stable_sparse_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, byteorder="little", signed=False) % 2_147_483_647) + 1


def _tokenize_for_sparse(text: str, *, max_tokens: int) -> list[str]:
    content = str(text or "").strip().lower()
    if not content:
        return []
    raw_tokens: list[str] = []
    for item in _TOKEN_PATTERN.findall(content):
        token = str(item or "").strip()
        if not token:
            continue
        if _CONTAINS_CJK_PATTERN.search(token):
            if len(token) == 1:
                raw_tokens.append(token)
            else:
                raw_tokens.append(token)
                raw_tokens.extend(token[idx : idx + 2] for idx in range(0, max(1, len(token) - 1)))
            continue
        if len(token) < 2:
            continue
        raw_tokens.append(token)
    if len(raw_tokens) <= max_tokens:
        return raw_tokens
    return raw_tokens[:max_tokens]


def _build_sparse_vector(text: str, *, max_tokens: int) -> Any | None:
    if SparseVector is None:
        return None
    tokens = _tokenize_for_sparse(text, max_tokens=max(16, int(max_tokens)))
    if not tokens:
        return None
    counts = Counter(tokens)
    weighted: dict[int, float] = {}
    for token, cnt in counts.items():
        idx = _stable_sparse_index(token)
        weighted[idx] = weighted.get(idx, 0.0) + float(cnt)
    if not weighted:
        return None
    pairs = sorted(weighted.items(), key=lambda item: item[0])
    indices = [idx for idx, _ in pairs]
    values = [weight for _, weight in pairs]
    return SparseVector(indices=indices, values=values)


_TOKEN_PATTERN = re.compile(r"[a-z0-9_./:-]+|[\u4e00-\u9fff]+", flags=re.IGNORECASE)
_CONTAINS_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_DEFAULT_DENSE_VECTOR_NAME = "dense"
_DEFAULT_SPARSE_VECTOR_NAME = "bm25"
_DEFAULT_BM25_MAX_TOKENS = 128
_DEFAULT_RRF_K = 60
_DEFAULT_HYBRID_CANDIDATE_FACTOR = 3
_DEFAULT_RERANK_ENABLED = True
_DEFAULT_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_DEFAULT_RERANK_TOP_N = 20
_DEFAULT_RERANK_BATCH_SIZE = 8
_DEFAULT_RERANK_MAX_LENGTH = 512

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QdrantConfig:
    url: str
    # 兼容历史单库名称（legacy fallback）。
    collection_name: str
    # 新增：分库名称。
    domain_collection_name: str
    case_collection_name: str
    vector_dim: int
    timeout_sec: float
    dense_vector_name: str
    sparse_vector_name: str
    bm25_max_tokens: int
    rrf_k: int
    hybrid_candidate_factor: int
    rerank_enabled: bool
    rerank_model_name: str
    rerank_top_n: int
    rerank_batch_size: int
    rerank_max_length: int
    rerank_local_only: bool

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        url = str(os.getenv("QDRANT_URL", "http://127.0.0.1:6333")).strip() or "http://127.0.0.1:6333"
        legacy_collection_name = (
            str(os.getenv("QDRANT_COLLECTION_NAME", "ai_ops_rag")).strip() or "ai_ops_rag"
        )
        domain_collection_name = (
            str(os.getenv("QDRANT_DOMAIN_COLLECTION_NAME", f"{legacy_collection_name}_domain")).strip()
            or f"{legacy_collection_name}_domain"
        )
        case_collection_name = (
            str(os.getenv("QDRANT_CASE_COLLECTION_NAME", f"{legacy_collection_name}_case")).strip()
            or f"{legacy_collection_name}_case"
        )
        vector_dim = max(1, _to_int(os.getenv("QDRANT_VECTOR_DIM", "512"), 512))
        timeout_sec = _to_float(os.getenv("QDRANT_TIMEOUT_SEC", "3"), 3.0)
        dense_vector_name = (
            str(os.getenv("QDRANT_DENSE_VECTOR_NAME", _DEFAULT_DENSE_VECTOR_NAME)).strip() or _DEFAULT_DENSE_VECTOR_NAME
        )
        sparse_vector_name = (
            str(os.getenv("QDRANT_SPARSE_VECTOR_NAME", _DEFAULT_SPARSE_VECTOR_NAME)).strip()
            or _DEFAULT_SPARSE_VECTOR_NAME
        )
        bm25_max_tokens = max(16, _to_int(os.getenv("QDRANT_BM25_MAX_TOKENS", _DEFAULT_BM25_MAX_TOKENS), _DEFAULT_BM25_MAX_TOKENS))
        rrf_k = max(1, _to_int(os.getenv("QDRANT_RRF_K", _DEFAULT_RRF_K), _DEFAULT_RRF_K))
        hybrid_candidate_factor = max(
            1,
            _to_int(
                os.getenv("QDRANT_HYBRID_CANDIDATE_FACTOR", _DEFAULT_HYBRID_CANDIDATE_FACTOR),
                _DEFAULT_HYBRID_CANDIDATE_FACTOR,
            ),
        )
        rerank_enabled = _to_bool(os.getenv("QDRANT_RERANK_ENABLED", str(int(_DEFAULT_RERANK_ENABLED))), _DEFAULT_RERANK_ENABLED)
        rerank_model_name = (
            str(os.getenv("QDRANT_RERANK_MODEL_NAME", _DEFAULT_RERANK_MODEL_NAME)).strip() or _DEFAULT_RERANK_MODEL_NAME
        )
        rerank_top_n = max(1, _to_int(os.getenv("QDRANT_RERANK_TOP_N", _DEFAULT_RERANK_TOP_N), _DEFAULT_RERANK_TOP_N))
        rerank_batch_size = max(
            1,
            _to_int(os.getenv("QDRANT_RERANK_BATCH_SIZE", _DEFAULT_RERANK_BATCH_SIZE), _DEFAULT_RERANK_BATCH_SIZE),
        )
        rerank_max_length = max(
            64,
            _to_int(os.getenv("QDRANT_RERANK_MAX_LENGTH", _DEFAULT_RERANK_MAX_LENGTH), _DEFAULT_RERANK_MAX_LENGTH),
        )
        rerank_local_only = _to_bool(
            os.getenv("QDRANT_RERANK_LOCAL_ONLY", os.getenv("AIOPS_BGE_LOCAL_ONLY", "1")),
            True,
        )
        return cls(
            url=url,
            collection_name=legacy_collection_name,
            domain_collection_name=domain_collection_name,
            case_collection_name=case_collection_name,
            vector_dim=vector_dim,
            timeout_sec=max(0.1, timeout_sec),
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            bm25_max_tokens=bm25_max_tokens,
            rrf_k=rrf_k,
            hybrid_candidate_factor=hybrid_candidate_factor,
            rerank_enabled=rerank_enabled,
            rerank_model_name=rerank_model_name,
            rerank_top_n=rerank_top_n,
            rerank_batch_size=rerank_batch_size,
            rerank_max_length=rerank_max_length,
            rerank_local_only=rerank_local_only,
        )


class QdrantStore:
    """Simple upsert/search wrapper for local Qdrant."""

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config = config or QdrantConfig.from_env()
        self._client: Any | None = None
        self._collection_profile_cache: dict[str, dict[str, Any]] = {}

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if QdrantClient is None or Distance is None or VectorParams is None:
            raise RuntimeError("qdrant_client is required. Install dependency: qdrant-client")
        self._client = QdrantClient(url=self.config.url, timeout=self.config.timeout_sec)
        return self._client

    def ensure_collection(self, *, collection_name: str | None = None) -> None:
        client = self._ensure_client()
        collections = client.get_collections()
        names = {row.name for row in list(collections.collections or [])}
        final_collection = str(collection_name or self.config.collection_name).strip()
        if not final_collection:
            raise ValueError("collection_name must not be empty")
        if final_collection in names:
            return
        if SparseVectorParams is not None and Modifier is not None:
            client.create_collection(
                collection_name=final_collection,
                vectors_config={
                    self.config.dense_vector_name: VectorParams(size=self.config.vector_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    self.config.sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF),
                },
            )
        else:
            client.create_collection(
                collection_name=final_collection,
                vectors_config=VectorParams(size=self.config.vector_dim, distance=Distance.COSINE),
            )
        self._collection_profile_cache.pop(final_collection, None)

    def _extract_points(self, response: Any) -> list[Any]:
        if response is None:
            return []
        return list(getattr(response, "points", None) or [])

    def _rows_from_points(self, points: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in list(points or []):
            row_id = getattr(row, "id", None)
            if row_id is None and isinstance(row, dict):
                row_id = row.get("id")
            row_score = getattr(row, "score", 0.0)
            if row_score is None and isinstance(row, dict):
                row_score = row.get("score", 0.0)
            payload = getattr(row, "payload", None)
            if payload is None and isinstance(row, dict):
                payload = row.get("payload")
            rows.append(
                {
                    "id": row_id,
                    "score": float(row_score or 0.0),
                    "payload": dict(payload or {}),
                }
            )
        return rows

    def _resolve_collection_profile(self, collection_name: str) -> dict[str, Any]:
        cached = self._collection_profile_cache.get(collection_name)
        if cached is not None:
            return dict(cached)
        profile = {
            "dense_using": None,
            "sparse_using": self.config.sparse_vector_name,
            "supports_sparse": False,
        }
        client = self._ensure_client()
        try:
            info = client.get_collection(collection_name=collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors_cfg = getattr(params, "vectors", None)
            if isinstance(vectors_cfg, dict):
                if self.config.dense_vector_name in vectors_cfg:
                    profile["dense_using"] = self.config.dense_vector_name
                elif vectors_cfg:
                    profile["dense_using"] = str(next(iter(vectors_cfg.keys())))
            sparse_cfg = getattr(params, "sparse_vectors", None)
            if isinstance(sparse_cfg, dict) and sparse_cfg:
                if self.config.sparse_vector_name in sparse_cfg:
                    profile["sparse_using"] = self.config.sparse_vector_name
                else:
                    profile["sparse_using"] = str(next(iter(sparse_cfg.keys())))
                profile["supports_sparse"] = True
        except Exception:
            # 读取 collection profile 失败时回退为 legacy unnamed dense。
            pass
        self._collection_profile_cache[collection_name] = dict(profile)
        return dict(profile)

    def _build_upsert_vector(self, *, collection_name: str, dense_vector: list[float], sparse_vector: Any | None) -> Any:
        profile = self._resolve_collection_profile(collection_name)
        dense_using = str(profile.get("dense_using") or "").strip()
        supports_sparse = bool(profile.get("supports_sparse"))
        sparse_using = str(profile.get("sparse_using") or self.config.sparse_vector_name).strip()
        if dense_using:
            vector_map: dict[str, Any] = {dense_using: dense_vector}
            if supports_sparse and sparse_vector is not None:
                vector_map[sparse_using] = sparse_vector
            return vector_map
        return dense_vector

    def _search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        client = self._ensure_client()
        max_limit = max(1, int(limit))
        if hasattr(client, "query_points"):
            profile = self._resolve_collection_profile(collection_name)
            dense_using = profile.get("dense_using")
            try:
                response = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=dense_using,
                    limit=max_limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                response = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=max_limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            return self._rows_from_points(self._extract_points(response))
        if hasattr(client, "search"):
            points = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=max_limit,
                score_threshold=score_threshold,
            )
            return self._rows_from_points(list(points or []))
        raise RuntimeError("Qdrant client has neither query_points nor search")

    def _search_sparse(self, *, collection_name: str, sparse_query: Any, limit: int) -> list[dict[str, Any]]:
        if sparse_query is None:
            return []
        client = self._ensure_client()
        if not hasattr(client, "query_points"):
            return []
        profile = self._resolve_collection_profile(collection_name)
        if not profile.get("supports_sparse"):
            return []
        sparse_using = str(profile.get("sparse_using") or self.config.sparse_vector_name).strip()
        if not sparse_using:
            return []
        try:
            response = client.query_points(
                collection_name=collection_name,
                query=sparse_query,
                using=sparse_using,
                limit=max(1, int(limit)),
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []
        return self._rows_from_points(self._extract_points(response))

    def _search_hybrid_by_qdrant_fusion(
        self,
        *,
        collection_name: str,
        dense_query: list[float],
        sparse_query: Any,
        limit: int,
        score_threshold: float | None = None,
        candidate_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if (
            sparse_query is None
            or Prefetch is None
            or FusionQuery is None
            or Fusion is None
        ):
            return []
        client = self._ensure_client()
        if not hasattr(client, "query_points"):
            return []
        profile = self._resolve_collection_profile(collection_name)
        if not profile.get("supports_sparse"):
            return []
        dense_using = profile.get("dense_using")
        sparse_using = str(profile.get("sparse_using") or self.config.sparse_vector_name).strip()
        prefetch_limit = max(max(1, int(limit)), int(candidate_limit or 0))
        dense_prefetch = Prefetch(
            query=dense_query,
            using=dense_using,
            limit=prefetch_limit,
            score_threshold=score_threshold,
        )
        sparse_prefetch = Prefetch(
            query=sparse_query,
            using=sparse_using,
            limit=prefetch_limit,
        )
        response = client.query_points(
            collection_name=collection_name,
            prefetch=[dense_prefetch, sparse_prefetch],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=max(1, int(limit)),
            with_payload=True,
            with_vectors=False,
        )
        return self._rows_from_points(self._extract_points(response))

    def _rrf_fuse_rows(
        self,
        *,
        dense_rows: list[dict[str, Any]],
        sparse_rows: list[dict[str, Any]],
        limit: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for source, rows in (("dense", dense_rows), ("bm25", sparse_rows)):
            for rank, row in enumerate(list(rows or []), start=1):
                raw_id = row.get("id")
                if raw_id is None:
                    continue
                doc_id = str(raw_id)
                item = fused.get(doc_id)
                if item is None:
                    item = {
                        "id": raw_id,
                        "score": 0.0,
                        "payload": dict(row.get("payload") or {}),
                        "dense_score": 0.0,
                        "bm25_score": 0.0,
                    }
                    fused[doc_id] = item
                item["score"] = float(item.get("score") or 0.0) + (1.0 / float(max(1, rrf_k) + rank))
                if source == "dense":
                    item["dense_score"] = float(row.get("score") or 0.0)
                else:
                    item["bm25_score"] = float(row.get("score") or 0.0)
        rows = list(fused.values())
        rows.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("dense_score") or 0.0),
                float(item.get("bm25_score") or 0.0),
            ),
            reverse=True,
        )
        return rows[: max(1, int(limit))]

    def _rerank_rows(self, *, query: str, rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        top_k = max(1, int(limit))
        if not self.config.rerank_enabled:
            return list(rows[:top_k])

        candidate_n = min(len(rows), max(top_k, int(self.config.rerank_top_n)))
        candidates = list(rows[:candidate_n])
        docs: list[str] = []
        for row in candidates:
            payload = dict(row.get("payload") or {})
            text = str(payload.get("text") or payload.get("content") or "").strip()
            docs.append(text)
        if not any(docs):
            return list(rows[:top_k])

        rerank_scores = rerank_documents(
            query=str(query or "").strip(),
            documents=docs,
            model_name=self.config.rerank_model_name,
            local_only=self.config.rerank_local_only,
            max_length=self.config.rerank_max_length,
            batch_size=self.config.rerank_batch_size,
        )
        if len(rerank_scores) != len(candidates):
            _LOGGER.warning(
                "rerank skipped due to score size mismatch: candidates=%d scores=%d",
                len(candidates),
                len(rerank_scores),
            )
            return list(rows[:top_k])

        reranked_rows: list[dict[str, Any]] = []
        for item, rerank_score in zip(candidates, rerank_scores):
            row = dict(item)
            row["rrf_score"] = float(row.get("score") or 0.0)
            row["rerank_score"] = float(rerank_score)
            row["score"] = float(rerank_score)
            reranked_rows.append(row)
        reranked_rows.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("rrf_score") or 0.0),
            ),
            reverse=True,
        )
        tail_rows = list(rows[candidate_n:])
        return [*reranked_rows, *tail_rows][:top_k]

    def collection_name_for_knowledge_type(self, knowledge_type: str) -> str:
        value = str(knowledge_type or "").strip().lower()
        if value == "case":
            return self.config.case_collection_name
        if value == "domain":
            return self.config.domain_collection_name
        return self.config.collection_name

    def upsert_text(
        self,
        *,
        text: str,
        point_id: str | int | None = None,
        payload: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> str | int:
        content = str(text or "").strip()
        if not content:
            raise ValueError("text must not be empty")
        final_collection = str(collection_name or self.config.collection_name).strip()
        self.ensure_collection(collection_name=final_collection)
        client = self._ensure_client()
        pid: str | int = point_id if point_id is not None else uuid4().hex
        dense_vector = text_embedding(content, dim=self.config.vector_dim)
        sparse_vector = _build_sparse_vector(content, max_tokens=self.config.bm25_max_tokens)
        point_payload = {"text": content, **dict(payload or {})}
        vector = self._build_upsert_vector(
            collection_name=final_collection,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        client.upsert(
            collection_name=final_collection,
            points=[PointStruct(id=pid, vector=vector, payload=point_payload)],
        )
        return pid

    def upsert_texts(
        self,
        items: list[dict[str, Any]],
        *,
        collection_name: str | None = None,
    ) -> list[str | int]:
        final_collection = str(collection_name or self.config.collection_name).strip()
        self.ensure_collection(collection_name=final_collection)
        client = self._ensure_client()
        rows: list[Any] = []
        ids: list[str | int] = []
        for item in list(items or []):
            content = str(item.get("text") or "").strip()
            if not content:
                continue
            pid: str | int = item.get("id") if item.get("id") is not None else uuid4().hex
            dense_vector = text_embedding(content, dim=self.config.vector_dim)
            sparse_vector = _build_sparse_vector(content, max_tokens=self.config.bm25_max_tokens)
            vector = self._build_upsert_vector(
                collection_name=final_collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            )
            payload = dict(item.get("payload") or {})
            rows.append(PointStruct(id=pid, vector=vector, payload={"text": content, **payload}))
            ids.append(pid)
        if rows:
            client.upsert(collection_name=final_collection, points=rows)
        return ids

    def search(
        self,
        *,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        final_collection = str(collection_name or self.config.collection_name).strip()
        self.ensure_collection(collection_name=final_collection)
        query_vector = text_embedding(query_text, dim=self.config.vector_dim)
        return self._search_dense(
            collection_name=final_collection,
            query_vector=query_vector,
            limit=max(1, int(limit)),
            score_threshold=score_threshold,
        )

    def search_hybrid(
        self,
        *,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        collection_name: str | None = None,
        semantic_limit: int | None = None,
        bm25_limit: int | None = None,
        rrf_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        final_collection = str(collection_name or self.config.collection_name).strip()
        self.ensure_collection(collection_name=final_collection)

        top_k = max(1, int(limit))
        candidate_factor = max(1, int(self.config.hybrid_candidate_factor))
        dense_limit = max(top_k, int(semantic_limit or (top_k * candidate_factor)))
        sparse_limit = max(top_k, int(bm25_limit or (top_k * candidate_factor)))
        dense_query = text_embedding(query_text, dim=self.config.vector_dim)
        sparse_query = _build_sparse_vector(query_text, max_tokens=self.config.bm25_max_tokens)

        # 优先使用 Qdrant 原生 fusion+RRF。
        try:
            fused_rows = self._search_hybrid_by_qdrant_fusion(
                collection_name=final_collection,
                dense_query=dense_query,
                sparse_query=sparse_query,
                limit=top_k,
                score_threshold=score_threshold,
                candidate_limit=max(dense_limit, sparse_limit),
            )
            if fused_rows:
                return self._rerank_rows(query=query_text, rows=fused_rows, limit=top_k)
        except Exception:
            # 失败时回退本地 RRF，避免影响主链路。
            pass

        dense_rows = self._search_dense(
            collection_name=final_collection,
            query_vector=dense_query,
            limit=dense_limit,
            score_threshold=score_threshold,
        )
        sparse_rows = self._search_sparse(
            collection_name=final_collection,
            sparse_query=sparse_query,
            limit=sparse_limit,
        )
        if sparse_rows:
            fused_rows = self._rrf_fuse_rows(
                dense_rows=dense_rows,
                sparse_rows=sparse_rows,
                limit=top_k,
                rrf_k=int(rrf_k or self.config.rrf_k),
            )
            if fused_rows:
                return self._rerank_rows(query=query_text, rows=fused_rows, limit=top_k)
        return self._rerank_rows(query=query_text, rows=dense_rows, limit=top_k)
