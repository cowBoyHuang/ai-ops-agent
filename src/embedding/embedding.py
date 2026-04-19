"""Text embedding and cosine similarity utilities."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

_BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_DEFAULT_EMBEDDING_DIM = 512
_LOCAL_FILES_ONLY = str(os.getenv("AIOPS_BGE_LOCAL_ONLY", "1")).strip() != "0"

try:  # Optional runtime dependency
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

_BGE_MODEL: Any | None = None
_BGE_TOKENIZER: Any | None = None
_MODEL_LOAD_TRIED = False


def _fit_dim(vector: list[float], dim: int) -> list[float]:
    if len(vector) == dim:
        return vector
    if len(vector) > dim:
        return vector[:dim]
    return [*vector, *([0.0] * (dim - len(vector)))]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [item / norm for item in vector]


def _hash_embedding(text: str, dim: int) -> list[float]:
    tokens = [item for item in text.split() if item] or list(text)
    vector = [0.0] * dim
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        idx = value % dim
        sign = 1.0 if ((value >> 9) & 1) == 0 else -1.0
        vector[idx] += sign
    return _l2_normalize(vector)


def _load_bge_model() -> bool:
    global _BGE_MODEL, _BGE_TOKENIZER, _MODEL_LOAD_TRIED
    if _MODEL_LOAD_TRIED:
        return _BGE_MODEL is not None and _BGE_TOKENIZER is not None
    _MODEL_LOAD_TRIED = True

    if AutoTokenizer is None or AutoModel is None or torch is None:
        return False
    try:
        _BGE_TOKENIZER = AutoTokenizer.from_pretrained(
            _BGE_MODEL_NAME,
            local_files_only=_LOCAL_FILES_ONLY,
        )
        _BGE_MODEL = AutoModel.from_pretrained(
            _BGE_MODEL_NAME,
            local_files_only=_LOCAL_FILES_ONLY,
        )
        _BGE_MODEL.eval()
        return True
    except Exception:
        _BGE_MODEL = None
        _BGE_TOKENIZER = None
        return False


def text_embedding(text: str, *, dim: int = _DEFAULT_EMBEDDING_DIM) -> list[float]:
    content = str(text or "").strip()
    target_dim = max(1, int(dim))
    if not content:
        return [0.0] * target_dim

    if _load_bge_model():
        assert _BGE_MODEL is not None
        assert _BGE_TOKENIZER is not None
        with torch.no_grad():
            encoded = _BGE_TOKENIZER(
                content,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = _BGE_MODEL(**encoded)
            vector = [float(item) for item in output.last_hidden_state[:, 0, :].squeeze(0).tolist()]
        return _fit_dim(_l2_normalize(vector), target_dim)

    return _hash_embedding(content, target_dim)


def cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = 0.0
    norm_lhs = 0.0
    norm_rhs = 0.0
    for l_item, r_item in zip(lhs, rhs):
        dot += l_item * r_item
        norm_lhs += l_item * l_item
        norm_rhs += r_item * r_item
    if norm_lhs <= 0.0 or norm_rhs <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_lhs) * math.sqrt(norm_rhs))
