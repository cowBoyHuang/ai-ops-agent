# AIOps Agent Demo Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可运行的 AIOps Agent Demo，提供 HTTP API 和简单 Web 页面，打通日志/代码/知识库/RCA 主链路，并接入 1 个真实 LLM。

**Architecture:** 采用模块化单体架构，在一个 FastAPI 进程内拆分 API、Agent、Skills、Tools、RAG、Memory、Cache、Router、Trace、Evaluation 和 Observability。业务工具先使用 Mock Adapter，但接口按真实外部系统设计，确保后续可平滑替换。

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, httpx, Jinja2, pytest, anyio, uvicorn

---

> 说明：当前工作目录未检测到 `.git`。以下 commit 步骤保留为执行要求，但实际执行前需要先初始化 git 仓库，或在真实仓库根目录中执行。

## File Structure

### Runtime

- Create: `pyproject.toml`
- Create: `.env`
- Create: `src/__init__.py`
- Create: `src/main.py`
- Create: `src/config.py`

### API

- Create: `src/api/schemas.py`
- Create: `src/api/routes/chat.py`
- Create: `src/api/routes/analyze.py`
- Create: `src/api/routes/sessions.py`

### Domain And State

- Create: `src/domain/enums.py`
- Create: `src/domain/models.py`
- Create: `src/session/store.py`
- Create: `src/memory/session_memory.py`
- Create: `src/cache/query_cache.py`

### Agent

- Create: `src/agent/planner.py`
- Create: `src/agent/orchestrator.py`
- Create: `src/agent/skills/base.py`
- Create: `src/agent/skills/knowledge.py`
- Create: `src/agent/skills/logs.py`
- Create: `src/agent/skills/code.py`
- Create: `src/agent/skills/rca.py`

### LLM

- Create: `src/llm/base.py`
- Create: `src/llm/openai_compatible.py`
- Create: `src/llm/router.py`
- Create: `src/llm/prompts.py`

### Retrieval And Tools

- Create: `src/rag/base.py`
- Create: `src/rag/keyword_retriever.py`
- Create: `src/tools/base.py`
- Create: `src/tools/mock_log_tool.py`
- Create: `src/tools/mock_code_tool.py`
- Create: `src/tools/fixture_loader.py`

### Platform

- Create: `src/tracing/models.py`
- Create: `src/tracing/collector.py`
- Create: `src/evaluation/evaluator.py`
- Create: `src/observability/metrics.py`

### Web

- Create: `src/web/templates/index.html`
- Create: `src/web/static/app.js`
- Create: `src/web/static/styles.css`

### Demo Data

- Create: `src/rag_ingest/source_docs/system_docs/order-service-architecture.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/db-pool-exhausted.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/downstream-500.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/validation-error.md`
- Create: `demo_data/logs/db-pool-exhausted.json`
- Create: `demo_data/logs/downstream-500.json`
- Create: `demo_data/logs/validation-error.json`
- Create: `demo_data/code/db-pool-exhausted.json`
- Create: `demo_data/code/downstream-500.json`
- Create: `demo_data/code/validation-error.json`

### Tests

- Create: `tests/conftest.py`
- Create: `tests/api/test_chat.py`
- Create: `tests/api/test_analyze.py`
- Create: `tests/domain/test_models.py`
- Create: `tests/session/test_store.py`
- Create: `tests/llm/test_router.py`
- Create: `tests/rag/test_keyword_retriever.py`
- Create: `tests/tools/test_mock_tools.py`
- Create: `tests/agent/test_planner.py`
- Create: `tests/agent/test_orchestrator.py`
- Create: `tests/evaluation/test_evaluator.py`

## Setup Preconditions

- [ ] 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Expected: `fastapi`、`pytest`、`httpx`、`uvicorn` 等依赖安装成功

## Chunk 1: Foundation And Contracts

### Task 1: Bootstrap FastAPI Service Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.env`
- Create: `src/__init__.py`
- Create: `src/main.py`
- Create: `src/config.py`
- Create: `src/api/routes/chat.py`
- Test: `tests/conftest.py`
- Test: `tests/api/test_chat.py`

- [ ] **Step 1: Write the failing health test**

```python
from fastapi.testclient import TestClient

from ai_ops_agent.main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/api/v1/chat")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_ops_agent'`

- [ ] **Step 3: Write minimal implementation**

```python
from fastapi import FastAPI

from ai_ops_agent.api.routes.health import router as chat_router

app = FastAPI(title="AIOps Agent Demo")
app.include_router(chat_router, prefix="/api/v1")
```

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_chat.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env src tests/api/test_chat.py tests/conftest.py
git commit -m "feat: bootstrap fastapi service"
```

### Task 2: Define Core Domain Models And Session Storage

**Files:**
- Create: `src/domain/enums.py`
- Create: `src/domain/models.py`
- Create: `src/session/store.py`
- Test: `tests/domain/test_models.py`
- Test: `tests/session/test_store.py`

- [ ] **Step 1: Write failing tests for request/session/result models**

```python
from ai_ops_agent.domain.models import AnalysisRequest, AnalysisResult


def test_analysis_request_defaults():
    request = AnalysisRequest(query="下单失败", service_name="order-service")
    assert request.environment == "prod"


def test_analysis_result_can_hold_root_cause():
    result = AnalysisResult(root_cause="数据库连接池耗尽", confidence=0.9)
    assert result.root_cause == "数据库连接池耗尽"
```

```python
from ai_ops_agent.session.store import InMemorySessionStore


def test_session_store_round_trip():
    store = InMemorySessionStore()
    store.save("s-1", {"status": "running"})
    assert store.get("s-1") == {"status": "running"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/domain/test_models.py tests/session/test_store.py -v`
Expected: FAIL with missing models and store implementations

- [ ] **Step 3: Write minimal implementation**

```python
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    query: str
    service_name: str
    environment: str = "prod"
    time_range: str = "15m"
```

```python
class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def save(self, session_id: str, payload: dict) -> None:
        self._data[session_id] = payload

    def get(self, session_id: str) -> dict | None:
        return self._data.get(session_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/domain/test_models.py tests/session/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain src/session tests/domain tests/session
git commit -m "feat: add core analysis models and session store"
```

### Task 3: Add LLM Client Contract And Router

**Files:**
- Create: `src/llm/base.py`
- Create: `src/llm/openai_compatible.py`
- Create: `src/llm/router.py`
- Create: `src/llm/prompts.py`
- Modify: `src/config.py`
- Test: `tests/llm/test_router.py`

- [ ] **Step 1: Write failing tests for router selection**

```python
from ai_ops_agent.llm.router import ModelRouter


def test_router_returns_planner_model():
    router = ModelRouter(default_model="gpt-4.1-mini", planner_model="gpt-4.1")
    assert router.for_task("planner") == "gpt-4.1"
    assert router.for_task("rca") == "gpt-4.1-mini"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llm/test_router.py -v`
Expected: FAIL with missing router implementation

- [ ] **Step 3: Write minimal implementation**

```python
class ModelRouter:
    def __init__(self, default_model: str, planner_model: str | None = None) -> None:
        self.default_model = default_model
        self.planner_model = planner_model or default_model

    def for_task(self, task: str) -> str:
        if task == "planner":
            return self.planner_model
        return self.default_model
```

```python
class OpenAICompatibleClient:
    async def generate(self, model: str, messages: list[dict]) -> dict:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/llm/test_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm src/config.py tests/llm/test_router.py
git commit -m "feat: add llm router and provider contract"
```

### Task 4: Add Retrieval, Memory, And Cache Interfaces

**Files:**
- Create: `src/rag/base.py`
- Create: `src/rag/keyword_retriever.py`
- Create: `src/memory/session_memory.py`
- Create: `src/cache/query_cache.py`
- Test: `tests/rag/test_keyword_retriever.py`

- [ ] **Step 1: Write failing retrieval test**

```python
from ai_ops_agent.rag.keyword_retriever import KeywordRetriever


def test_keyword_retriever_returns_matching_document(tmp_path):
    docs = [("doc-1", "数据库连接池耗尽会导致下单失败")]
    retriever = KeywordRetriever(documents=docs)
    result = retriever.search("下单失败 数据库")
    assert result[0].document_id == "doc-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_keyword_retriever.py -v`
Expected: FAIL with missing retriever implementation

- [ ] **Step 3: Write minimal implementation**

```python
class KeywordRetriever:
    def __init__(self, documents: list[tuple[str, str]]) -> None:
        self.documents = documents

    def search(self, query: str) -> list:
        tokens = set(query.split())
        ranked = []
        for document_id, content in self.documents:
            score = sum(1 for token in tokens if token in content)
            if score:
                ranked.append(RetrievedDocument(document_id=document_id, score=score, content=content))
        return sorted(ranked, key=lambda item: item.score, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/rag/test_keyword_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag src/memory src/cache tests/rag/test_keyword_retriever.py
git commit -m "feat: add rag memory and cache interfaces"
```

## Chunk 2: Analysis Pipeline

### Task 5: Add Mock Tool Adapters And Fixture Data

**Files:**
- Create: `src/tools/base.py`
- Create: `src/tools/fixture_loader.py`
- Create: `src/tools/mock_log_tool.py`
- Create: `src/tools/mock_code_tool.py`
- Create: `demo_data/logs/db-pool-exhausted.json`
- Create: `demo_data/logs/downstream-500.json`
- Create: `demo_data/logs/validation-error.json`
- Create: `demo_data/code/db-pool-exhausted.json`
- Create: `demo_data/code/downstream-500.json`
- Create: `demo_data/code/validation-error.json`
- Test: `tests/tools/test_mock_tools.py`

- [ ] **Step 1: Write failing tests for mock tools**

```python
from ai_ops_agent.tools.mock_log_tool import MockLogTool


def test_mock_log_tool_returns_fixture():
    tool = MockLogTool(fixtures_dir="demo_data/logs")
    result = tool.query(service_name="order-service", symptom="下单失败")
    assert result.items
    assert result.items[0]["source"] == "mock-log"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_mock_tools.py -v`
Expected: FAIL with missing tool implementations or missing fixture files

- [ ] **Step 3: Write minimal implementation**

```python
class MockLogTool:
    def __init__(self, fixtures_dir: str) -> None:
        self.fixtures_dir = fixtures_dir

    def query(self, service_name: str, symptom: str) -> ToolResult:
        payload = load_case_fixture(self.fixtures_dir, symptom)
        return ToolResult(source="mock-log", items=payload["items"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_mock_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools demo_data tests/tools/test_mock_tools.py
git commit -m "feat: add mock tool adapters and fixtures"
```

### Task 6: Build Planner And Skill Contracts

**Files:**
- Create: `src/agent/planner.py`
- Create: `src/agent/skills/base.py`
- Create: `src/agent/skills/knowledge.py`
- Create: `src/agent/skills/logs.py`
- Create: `src/agent/skills/code.py`
- Test: `tests/agent/test_planner.py`

- [ ] **Step 1: Write failing tests for default plan**

```python
from ai_ops_agent.agent.planner import DefaultPlanner
from ai_ops_agent.domain.models import AnalysisRequest


def test_default_planner_builds_four_steps():
    planner = DefaultPlanner()
    request = AnalysisRequest(query="下单失败", service_name="order-service")
    plan = planner.build(request)
    assert [step.name for step in plan.steps] == [
        "knowledge_search",
        "log_analysis",
        "code_search",
        "rca_synthesis",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_planner.py -v`
Expected: FAIL with missing planner implementation

- [ ] **Step 3: Write minimal implementation**

```python
class DefaultPlanner:
    def build(self, request: AnalysisRequest) -> AnalysisPlan:
        return AnalysisPlan(
            steps=[
                PlanStep(step_id="1", name="knowledge_search", goal="检索知识库"),
                PlanStep(step_id="2", name="log_analysis", goal="分析日志"),
                PlanStep(step_id="3", name="code_search", goal="检索代码"),
                PlanStep(step_id="4", name="rca_synthesis", goal="汇总结论"),
            ]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/planner.py src/agent/skills tests/agent/test_planner.py
git commit -m "feat: add planner and skill contracts"
```

### Task 7: Build Orchestrator And RCA Skill

**Files:**
- Create: `src/agent/orchestrator.py`
- Create: `src/agent/skills/rca.py`
- Modify: `src/llm/prompts.py`
- Test: `tests/agent/test_orchestrator.py`

- [ ] **Step 1: Write failing orchestrator test**

```python
from ai_ops_agent.agent.orchestrator import AnalysisOrchestrator
from ai_ops_agent.domain.models import AnalysisRequest


def test_orchestrator_returns_result(fake_dependencies):
    orchestrator = AnalysisOrchestrator(**fake_dependencies)
    result = orchestrator.run(AnalysisRequest(query="下单失败", service_name="order-service"))
    assert result.root_cause
    assert result.evidence
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_orchestrator.py -v`
Expected: FAIL with missing orchestrator or RCA synthesis

- [ ] **Step 3: Write minimal implementation**

```python
class AnalysisOrchestrator:
    def run(self, request: AnalysisRequest) -> AnalysisResult:
        plan = self.planner.build(request)
        artifacts = []
        for step in plan.steps[:-1]:
            artifacts.extend(self.skill_registry[step.name].execute(request))
        return self.skill_registry["rca_synthesis"].execute(request, artifacts)
```

```python
RCA_PROMPT_TEMPLATE = """
你是一名值班排障助手。基于 artifacts 和 evidence 输出 JSON：
{{
  "root_cause": "...",
  "confidence": 0.0,
  "suggestions": ["..."],
  "summary": "..."
}}
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/orchestrator.py src/agent/skills/rca.py src/llm/prompts.py tests/agent/test_orchestrator.py
git commit -m "feat: add orchestrator and rca synthesis"
```

### Task 8: Load Knowledge Base And Wire Retrieval Into Skills

**Files:**
- Create: `src/rag_ingest/source_docs/system_docs/order-service-architecture.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/db-pool-exhausted.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/downstream-500.md`
- Create: `src/rag_ingest/source_docs/troubleshooting/validation-error.md`
- Modify: `src/agent/skills/knowledge.py`
- Modify: `src/rag/keyword_retriever.py`
- Test: `tests/rag/test_keyword_retriever.py`

- [ ] **Step 1: Add failing test for markdown knowledge loading**

```python
from ai_ops_agent.rag.keyword_retriever import load_markdown_documents


def test_load_markdown_documents(tmp_path):
    doc = tmp_path / "case.md"
    doc.write_text("# title\n数据库连接池耗尽", encoding="utf-8")
    documents = load_markdown_documents(tmp_path)
    assert documents[0].content.startswith("# title")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/rag/test_keyword_retriever.py -v`
Expected: FAIL with missing loader or broken retrieval wiring

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


def load_markdown_documents(root: Path) -> list[RetrievedDocument]:
    documents = []
    for path in root.rglob("*.md"):
        documents.append(RetrievedDocument(document_id=path.stem, content=path.read_text(encoding="utf-8")))
    return documents
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/rag/test_keyword_retriever.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rag_ingest/source_docs src/agent/skills/knowledge.py src/rag tests/rag/test_keyword_retriever.py
git commit -m "feat: add knowledge fixtures and retrieval loading"
```

## Chunk 3: API, Web, And Platform Capabilities

### Task 9: Expose Analyze And Session APIs

**Files:**
- Create: `src/api/schemas.py`
- Create: `src/api/routes/analyze.py`
- Create: `src/api/routes/sessions.py`
- Modify: `src/main.py`
- Test: `tests/api/test_analyze.py`

- [ ] **Step 1: Write failing API test for analysis flow**

```python
from fastapi.testclient import TestClient

from ai_ops_agent.main import app


def test_analyze_returns_structured_result():
    client = TestClient(app)
    response = client.post(
        "/api/v1/analyze",
        json={"query": "下单失败", "service_name": "order-service", "environment": "prod"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["root_cause"]
    assert body["trace_summary"]["steps"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/api/test_analyze.py -v`
Expected: FAIL with missing analyze or sessions routes

- [ ] **Step 3: Write minimal implementation**

```python
@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, orchestrator: AnalysisOrchestrator = Depends(get_orchestrator)):
    session = orchestrator.run(request.to_domain())
    return AnalyzeResponse.from_session(session)
```

```python
@router.get("/sessions/{session_id}")
def get_session(session_id: str, store: InMemorySessionStore = Depends(get_session_store)):
    return store.get(session_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_analyze.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api src/main.py tests/api/test_analyze.py
git commit -m "feat: expose analyze and session apis"
```

### Task 10: Add Trace, Evaluation, And Metrics Collection

**Files:**
- Create: `src/tracing/models.py`
- Create: `src/tracing/collector.py`
- Create: `src/evaluation/evaluator.py`
- Create: `src/observability/metrics.py`
- Modify: `src/agent/orchestrator.py`
- Test: `tests/evaluation/test_evaluator.py`

- [ ] **Step 1: Write failing test for evaluation output**

```python
from ai_ops_agent.evaluation.evaluator import RuleBasedEvaluator
from ai_ops_agent.domain.models import AnalysisResult


def test_evaluator_scores_complete_result():
    evaluator = RuleBasedEvaluator()
    result = AnalysisResult(root_cause="数据库连接池耗尽", confidence=0.8, suggestions=["增加连接池"])
    score = evaluator.evaluate(result)
    assert score["completeness"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_evaluator.py -v`
Expected: FAIL with missing evaluator implementation

- [ ] **Step 3: Write minimal implementation**

```python
class RuleBasedEvaluator:
    def evaluate(self, result: AnalysisResult) -> dict[str, float]:
        return {
            "accuracy": 1.0 if result.root_cause else 0.0,
            "completeness": 1.0 if result.suggestions else 0.5,
            "actionability": 1.0 if result.suggestions else 0.0,
        }
```

```python
class TraceCollector:
    def record_step(self, step_name: str, status: str, latency_ms: int, output_summary: str) -> None:
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_evaluator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracing src/evaluation src/observability tests/evaluation/test_evaluator.py
git commit -m "feat: add trace evaluation and metrics"
```

### Task 11: Add Simple Web Demo Page

**Files:**
- Create: `src/web/templates/index.html`
- Create: `src/web/static/app.js`
- Create: `src/web/static/styles.css`
- Modify: `src/main.py`
- Test: `tests/api/test_chat.py`

- [ ] **Step 1: Add a failing test for root page**

```python
def test_root_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AIOps Agent Demo" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_chat.py -v`
Expected: FAIL with missing root route or missing template

- [ ] **Step 3: Write minimal implementation**

```html
<form id="analyze-form">
  <input name="service_name" value="order-service" />
  <textarea name="query">下单失败</textarea>
  <button type="submit">开始分析</button>
</form>
<section id="timeline"></section>
<section id="result"></section>
```

```javascript
document.getElementById("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/api/v1/analyze", { method: "POST", body: JSON.stringify(payload) });
  const data = await response.json();
  renderTimeline(data.trace_summary.steps);
  renderResult(data.result);
});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/api/test_chat.py tests/api/test_analyze.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/web src/main.py tests/api
git commit -m "feat: add demo web interface"
```

### Task 12: Add End-To-End Verification And Runbook

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_STRUCTURE.md`
- Test: `tests/api/test_analyze.py`
- Test: `tests/agent/test_orchestrator.py`

- [ ] **Step 1: Add one full happy-path integration test**

```python
def test_end_to_end_db_pool_case(client):
    response = client.post(
        "/api/v1/analyze",
        json={"query": "下单失败，日志里有 DBConnectionTimeout", "service_name": "order-service"},
    )
    body = response.json()
    assert "数据库连接池" in body["result"]["root_cause"]
    assert body["evaluation"]["actionability"] >= 0.5
```

- [ ] **Step 2: Run test to verify it fails until the full stack is wired**

Run: `pytest tests/api/test_analyze.py::test_end_to_end_db_pool_case -v`
Expected: FAIL until API, orchestrator, tools, and evaluator are fully connected

- [ ] **Step 3: Complete any missing wiring and document how to run**

```bash
python -m uvicorn ai_ops_agent.main:app --reload --app-dir src
pytest -v
```

README additions:

- environment variables
- how to start API
- how to open Web UI
- available demo scenarios

- [ ] **Step 4: Run the full verification suite**

Run: `pytest -v`
Expected: PASS for all tests

Run: `python -m uvicorn ai_ops_agent.main:app --reload --app-dir src`
Expected: service starts and `/docs` plus `/` are reachable

- [ ] **Step 5: Commit**

```bash
git add README.md docs/PROJECT_STRUCTURE.md tests
git commit -m "docs: add runbook and verify aiops demo flow"
```

## Execution Notes

- 优先保证结构化对象契约稳定，再做界面润色。
- 不要在第一期引入真实日志系统接入。
- `RAG` 第一版允许使用关键词检索，只要接口保持稳定即可。
- `LLMClient` 必须保留统一抽象，避免将供应商细节泄漏到 Skill 或 Orchestrator。
- 所有 Skill 和 Tool 输出都要带来源信息，供 Trace 和前端展示使用。
- 执行期间配合 `@superpowers:verification-before-completion` 做真实验证，不要在未运行测试前宣称完成。

## Handoff Checklist

- [ ] 所有任务按顺序执行
- [ ] 每个任务先写失败测试，再做最小实现
- [ ] 每个任务执行对应测试命令
- [ ] API `/api/v1/analyze`、`/api/v1/sessions/{id}`、`/api/v1/chat` 可用
- [ ] Web 页面 `/` 可提交分析请求并展示结果
- [ ] 真实 LLM 已通过配置接通
- [ ] 三个 Demo 场景可演示
- [ ] 全量测试通过
