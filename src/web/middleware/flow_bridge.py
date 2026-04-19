from __future__ import annotations

from flow.main import run as flow_main_run
from web.schemas import AnalyzeRequest, AnalyzeResponse


def _to_flow_payload(req: AnalyzeRequest) -> dict[str, object]:
    extra = dict(req.extra or {})
    return {
        "message": req.query,
        "query": req.query,
        "chat_id": req.chatId,
        "user_id": req.userId,
        "extra": extra,
        **extra,
    }


def _to_http_response(req: AnalyzeRequest, state: dict[str, object]) -> AnalyzeResponse:
    response = state.get("response")
    response_payload = dict(response) if isinstance(response, dict) else {}
    return AnalyzeResponse(
        chatId=str(response_payload.get("chatId") or state.get("chat_id") or req.chatId or ""),
        status=str(response_payload.get("status") or state.get("status") or "running"),
        message=str(response_payload.get("message") or state.get("error") or ""),
    )


def handle_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    state = flow_main_run(_to_flow_payload(req))
    return _to_http_response(req, state)
