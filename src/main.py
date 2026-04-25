from __future__ import annotations

from runtime_logging import configure_runtime_logging
from web.app import create_app

configure_runtime_logging()
app = create_app()
