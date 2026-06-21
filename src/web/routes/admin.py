from __future__ import annotations

from fastapi import APIRouter

from flow.modules.memory.memory import clear_persistent_data
from web.schemas import ClearStorageResponse

API_V1_PREFIX = "/api/v1"
router = APIRouter(prefix=API_V1_PREFIX, tags=["admin"])


@router.post("/admin/clear-storage", response_model=ClearStorageResponse, summary="Clear local DB and Redis data only")
def clear_storage() -> ClearStorageResponse:
    result = clear_persistent_data()
    return ClearStorageResponse(status="ok", **result)
