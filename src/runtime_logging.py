from __future__ import annotations

from contextvars import ContextVar, Token
import logging
from pathlib import Path

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("aiops_request_id", default="-")
_FACTORY_INSTALLED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def logs_dir() -> Path:
    path = _repo_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _install_record_factory() -> None:
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return

    old_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        record = old_factory(*args, **kwargs)
        record.request_id = _REQUEST_ID_CTX.get("-")
        return record

    logging.setLogRecordFactory(_factory)
    _FACTORY_INSTALLED = True


def configure_runtime_logging() -> Path:
    """初始化运行时日志基础能力（目录 + request_id 注入）。"""
    path = logs_dir()
    _install_record_factory()
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    return path


def bind_request_id(request_id: str) -> Token[str]:
    return _REQUEST_ID_CTX.set(str(request_id or "-"))


def reset_request_id(token: Token[str]) -> None:
    _REQUEST_ID_CTX.reset(token)


class _RequestIdFilter(logging.Filter):
    def __init__(self, request_id: str) -> None:
        super().__init__()
        self.request_id = str(request_id or "-")

    def filter(self, record: logging.LogRecord) -> bool:
        return str(getattr(record, "request_id", "-")) == self.request_id


def build_request_file_handler(request_id: str) -> logging.FileHandler:
    rid = str(request_id or "-")
    path = logs_dir() / f"{rid}.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(request_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_RequestIdFilter(rid))
    return file_handler
