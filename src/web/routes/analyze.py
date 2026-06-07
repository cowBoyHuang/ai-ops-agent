from __future__ import annotations

from fastapi import APIRouter

from web.middleware.flow_bridge import handle_analyze
from web.schemas import AnalyzeRequest, AnalyzeResponse

API_V1_PREFIX = "/api/v1"
ANALYZE_PATH = f"{API_V1_PREFIX}/analyze"

router = APIRouter(prefix=API_V1_PREFIX, tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyze issue evidence")
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return handle_analyze(req)
