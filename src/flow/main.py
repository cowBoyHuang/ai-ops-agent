from __future__ import annotations

import logging
import uuid
from typing import Any

from flow.flow import build_langchain_error_chain, build_langchain_main_chain
from runtime_logging import bind_request_id, build_request_file_handler, get_request_id, reset_request_id

_FLOW_LOG = logging.getLogger("aiops.flow")


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """Flow 中间层入口：主链路异常时切换到错误链路。"""
    context = dict(payload)
    pre = get_request_id()
    self_managed = pre in ("-", "")
    token = None
    handler: logging.Handler | None = None
    if self_managed:
        rid = str(context.get("request_id") or "").strip() or uuid.uuid4().hex
        context["request_id"] = rid
        token = bind_request_id(rid)
        handler = build_request_file_handler(rid)
        logging.getLogger().addHandler(handler)
    else:
        context.setdefault("request_id", pre)

    _FLOW_LOG.info("flow.start chat_id=%s", str(context.get("chat_id") or ""))
    try:
        result = build_langchain_main_chain().invoke(context)
        _FLOW_LOG.info("flow.end status=%s", str((result or {}).get("status") or ""))
        return result
    except Exception as exc:  # noqa: BLE001
        context["error"] = str(exc)
        _FLOW_LOG.exception("flow.error")
        err = build_langchain_error_chain().invoke(context)
        _FLOW_LOG.info("flow.end status=%s", str((err or {}).get("status") or "failed"))
        return err
    finally:
        if handler is not None and token is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
            reset_request_id(token)
