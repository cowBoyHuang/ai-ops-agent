from __future__ import annotations

from fastapi import FastAPI

from web.routes.analyze import router as analyze_router


def create_app() -> FastAPI:
    app = FastAPI(title="AIOps Agent")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(analyze_router)
    return app
