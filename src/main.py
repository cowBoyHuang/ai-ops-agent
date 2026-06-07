from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

# 必须在加载 .env 后再导入业务模块，避免导入期读取到空配置（如 REDIS_URL）。
from runtime_logging import configure_runtime_logging
from web.app import create_app

configure_runtime_logging()
app = create_app()


def main() -> None:
    host = os.getenv("AIOPS_HOST", "0.0.0.0")
    port = int(os.getenv("AIOPS_PORT", "8000"))
    reload_enabled = os.getenv("AIOPS_RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    uvicorn.run("main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
