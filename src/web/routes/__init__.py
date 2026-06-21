"""Web route modules."""

from web.routes.admin import router as admin_router
from web.routes.analyze import router as analyze_router

__all__ = ["analyze_router", "admin_router"]
