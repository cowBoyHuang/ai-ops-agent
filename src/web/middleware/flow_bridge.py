from __future__ import annotations

from flow.main import run as flow_main_run
from web.schemas import AnalyzeRequest, AnalyzeResponse


def _to_flow_payload(req: AnalyzeRequest) -> dict[str, object]:
    extra = dict(req.extra or {})
    question = str(req.question or "")
    chat_id = str(req.chat_id or "")
    user_id = str(req.user_id or "")
    return {
        "message": question,
        "query": question,
        "chat_id": chat_id,
        "user_id": user_id,
        "extra": extra,
        **extra,
    }


def _to_http_response(req: AnalyzeRequest, state: dict[str, object]) -> AnalyzeResponse:
    response = state.get("response")
    response_payload = dict(response) if isinstance(response, dict) else {}
    return AnalyzeResponse(
        chatId=str(response_payload.get("chatId") or state.get("chat_id") or req.chat_id or ""),
        status=str(response_payload.get("status") or state.get("status") or "running"),
        message=str(response_payload.get("message") or state.get("error") or ""),
    )


def handle_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    state = flow_main_run(_to_flow_payload(req))
    return _to_http_response(req, state)
