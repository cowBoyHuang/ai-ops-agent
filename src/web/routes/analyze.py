from __future__ import annotations

from fastapi import APIRouter

from web.middleware.flow_bridge import handle_analyze
from web.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return handle_analyze(req)
