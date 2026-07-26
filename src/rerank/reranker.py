"""Local cross-encoder reranker utilities."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_LOGGER = logging.getLogger(__name__)
_DEFAULT_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_DEFAULT_RERANK_MAX_LENGTH = 512
_DEFAULT_RERANK_BATCH_SIZE = 8

try:  # Optional runtime dependency.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple[Any, Any] | None] = {}


def _cache_key(model_name: str, local_only: bool) -> str:
    return f"{model_name}|local_only={int(bool(local_only))}"


def _load_rerank_model(model_name: str, *, local_only: bool) -> tuple[Any, Any] | None:
    key = _cache_key(model_name, local_only)
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = None
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_only)
        model.eval()
        loaded: tuple[Any, Any] | None = (tokenizer, model)
    except Exception as err:  # pragma: no cover - external dependency failure
        _LOGGER.warning("reranker model load failed: model=%s err=%s", model_name, err)
        loaded = None

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = loaded
    return loaded


def rerank_documents(
    *,
    query: str,
    documents: list[str],
    model_name: str | None = None,
    local_only: bool | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> list[float]:
    """Score each document with a local cross-encoder reranker.

    Returns:
    - list[float]: per-document logits (higher means more relevant).
      Returns [] when model is unavailable or scoring fails.
    """
    normalized_query = str(query or "").strip()
    normalized_docs = [str(item or "").strip() for item in list(documents or [])]
    if not normalized_query or not normalized_docs:
        return []

    final_model_name = str(model_name or os.getenv("QDRANT_RERANK_MODEL_NAME", _DEFAULT_RERANK_MODEL_NAME)).strip()
    if not final_model_name:
        final_model_name = _DEFAULT_RERANK_MODEL_NAME
    if local_only is None:
        local_only = str(os.getenv("QDRANT_RERANK_LOCAL_ONLY", "1")).strip() != "0"
    final_max_length = max(64, int(max_length or os.getenv("QDRANT_RERANK_MAX_LENGTH", _DEFAULT_RERANK_MAX_LENGTH)))
    final_batch_size = max(1, int(batch_size or os.getenv("QDRANT_RERANK_BATCH_SIZE", _DEFAULT_RERANK_BATCH_SIZE)))

    loaded = _load_rerank_model(final_model_name, local_only=bool(local_only))
    if loaded is None:
        return []
    tokenizer, model = loaded

    scores: list[float] = []
    try:
        with torch.no_grad():
            for start in range(0, len(normalized_docs), final_batch_size):
                batch_docs = normalized_docs[start : start + final_batch_size]
                batch_queries = [normalized_query] * len(batch_docs)
                encoded = tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=final_max_length,
                    return_tensors="pt",
                )
                output = model(**encoded)
                logits = output.logits
                if len(getattr(logits, "shape", ())) > 1 and int(logits.shape[-1]) == 1:
                    logits = logits.squeeze(-1)
                batch_scores = [float(item) for item in logits.detach().cpu().tolist()]
                if len(batch_scores) != len(batch_docs):
                    return []
                scores.extend(batch_scores)
    except Exception as err:  # pragma: no cover - model runtime failure
        _LOGGER.warning("reranker score failed: model=%s err=%s", final_model_name, err)
        return []

    if len(scores) != len(normalized_docs):
        return []
    return scores

