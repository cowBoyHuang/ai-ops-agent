from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from runtime_logging import configure_runtime_logging
from web.app import create_app

_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

configure_runtime_logging()
app = create_app()
