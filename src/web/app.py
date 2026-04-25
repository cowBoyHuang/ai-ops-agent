from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse

from runtime_logging import bind_request_id, build_request_file_handler, logs_dir, reset_request_id
from web.routes.analyze import router as analyze_router

_REQUEST_LOGGER = logging.getLogger("aiops.request")


def create_app() -> FastAPI:
    app = FastAPI(title="AIOps Agent")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = uuid.uuid4().hex
        token = bind_request_id(request_id)
        root_logger = logging.getLogger()
        request_handler = build_request_file_handler(request_id)
        root_logger.addHandler(request_handler)

        start = time.monotonic()
        method = request.method.upper()
        path = request.url.path
        query = str(request.url.query or "")
        body_preview = ""
        if method == "POST" and path == "/api/v1/analyze":
            try:
                raw = (await request.body()).decode("utf-8", errors="ignore")
                body_preview = raw[:1000]
            except Exception:  # noqa: BLE001
                body_preview = "<failed to read body>"
        _REQUEST_LOGGER.info("request.start method=%s path=%s query=%s body=%s", method, path, query, body_preview)
        try:
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            _REQUEST_LOGGER.info(
                "request.end method=%s path=%s status=%s duration_ms=%.2f",
                method,
                path,
                response.status_code,
                elapsed_ms,
            )
            response.headers["X-Request-Id"] = request_id
            return response
        except Exception:  # noqa: BLE001
            elapsed_ms = (time.monotonic() - start) * 1000
            _REQUEST_LOGGER.exception(
                "request.error method=%s path=%s duration_ms=%.2f",
                method,
                path,
                elapsed_ms,
            )
            raise
        finally:
            root_logger.removeHandler(request_handler)
            request_handler.close()
            reset_request_id(token)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>问题调试</title>
    <style>
      :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --text: #1d2a3b;
        --primary: #0b67ff;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
        background: var(--bg);
        color: var(--text);
      }
      .wrap {
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
      }
      .card {
        width: min(680px, 100%);
        background: var(--card);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 28px rgba(15, 34, 64, 0.1);
      }
      h1 { margin: 0 0 8px; font-size: 24px; }
      .field {
        display: grid;
        gap: 6px;
        margin: 10px 0;
      }
      input, textarea, button {
        font: inherit;
      }
      input, textarea {
        width: 100%;
        border: 1px solid #cdd9ef;
        border-radius: 8px;
        padding: 10px 12px;
        background: #fff;
      }
      textarea { min-height: 92px; resize: vertical; }
      button {
        margin-top: 6px;
        border: 0;
        border-radius: 8px;
        padding: 10px 14px;
        background: var(--primary);
        color: #fff;
        cursor: pointer;
      }
      pre {
        margin-top: 10px;
        background: #f6f9ff;
        border: 1px solid #d9e6ff;
        border-radius: 8px;
        padding: 10px;
        white-space: pre-wrap;
        word-break: break-word;
      }
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="card">
        <h1>问题调试</h1>
        <p style="margin:0 0 10px;color:#52637a;">请求日志目录：__LOGS_DIR__（单次请求一个日志文件）</p>
        <div class="field">
          <label for="chat_id">chat_id</label>
          <input id="chat_id" placeholder="例如: chat_123" />
        </div>
        <div class="field">
          <label for="user_id">user_id</label>
          <input id="user_id" placeholder="例如: u_123" />
        </div>
        <div class="field">
          <label for="question">question</label>
          <textarea id="question" placeholder="请输入问题，例如：订单创建失败，traceId=abc123"></textarea>
        </div>
        <button id="ask_btn" type="button">发送问答请求</button>
        <pre id="qa_result">等待请求...</pre>
      </section>
    </main>
    <script>
      const askBtn = document.getElementById("ask_btn");
      const resultBox = document.getElementById("qa_result");
      askBtn.addEventListener("click", async () => {
        const payload = {
          chat_id: document.getElementById("chat_id").value || "",
          user_id: document.getElementById("user_id").value || "",
          question: document.getElementById("question").value || ""
        };
        resultBox.textContent = "请求中...";
        try {
          const resp = await fetch("/api/v1/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const data = await resp.json();
          resultBox.textContent = JSON.stringify(data, null, 2);
        } catch (err) {
          resultBox.textContent = "请求失败: " + String(err);
        }
      });
    </script>
  </body>
</html>
""".replace("__LOGS_DIR__", str(logs_dir()))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(analyze_router)
    return app
