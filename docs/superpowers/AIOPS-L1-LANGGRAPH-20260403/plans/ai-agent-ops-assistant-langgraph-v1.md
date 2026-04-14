# AI Agent 运维助手（LangGraph v1）实施计划

> **给智能代理工作者：** 必需：使用 qsuperpowers:subagent-driven-development（如果有子代理可用）或 qsuperpowers:executing-plans 来执行此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 在现有 AIOps Demo 上落地 L1 诊断助手，采用 `Plan -> 循环执行(MCP/Skills/RAG) -> LLM Final` 动态重规划链路，并实现外部优先、失败可恢复、短期/长期记忆与经验向量化。

**架构：** 保持 FastAPI 单体部署，新增 LangGraph 状态机作为分析内核，Skill 层改为 Provider 适配（外部优先 + 本地 fallback）。记忆层拆分短期（会话内压缩）与长期（MySQL + 向量经验库），Final Synthesis 改为基于证据的定性置信等级输出。

**技术栈：** Python 3.11+, FastAPI, LangGraph, LlamaIndex, httpx, Pydantic v2, SQLAlchemy(MySQL), pytest

---

> 说明：当前目录未检测到 `.git`。以下 `git commit` 步骤保留为执行要求；若仍不在仓库根目录，执行时先切到真实 git root。

## 文件结构（本计划涉及）

### 新增文件

- `src/graph/state.py`
- `src/graph/nodes.py`
- `src/graph/workflow.py`
- `src/providers/logs/base.py`
- `src/providers/logs/external_http.py`
- `src/providers/logs/local_case.py`
- `src/providers/code/base.py`
- `src/providers/code/git_remote.py`
- `src/providers/code/local_repo.py`
- `src/rag/llamaindex_retriever.py`
- `src/memory/short_term.py`
- `src/memory/long_term_store.py`
- `src/memory/experience_store.py`
- `src/session/lifecycle.py`
- `tests/graph/test_judge_next.py`
- `tests/graph/test_workflow.py`
- `tests/providers/test_log_providers.py`
- `tests/providers/test_code_providers.py`
- `tests/rag/test_llamaindex_retriever.py`
- `tests/memory/test_short_term_memory.py`
- `tests/memory/test_long_term_store.py`
- `tests/memory/test_experience_store.py`
- `tests/api/test_sessions_lifecycle.py`

### 修改文件

- `pyproject.toml`
- `.env`
- `src/config.py`
- `src/domain/enums.py`
- `src/domain/models.py`
- `src/agent/planner.py`
- `src/agent/orchestrator.py`
- `src/agent/skills/logs.py`
- `src/agent/skills/code.py`
- `src/agent/skills/knowledge.py`
- `src/agent/skills/rca.py`
- `src/llm/openai_compatible.py`
- `src/llm/prompts.py`
- `src/main_support.py`
- `src/api/schemas.py`
- `src/api/routes/analyze.py`
- `src/api/routes/chat.py`
- `src/api/routes/sessions.py`
- `tests/domain/test_models.py`
- `tests/agent/test_planner.py`
- `tests/agent/test_orchestrator.py`
- `tests/api/test_analyze.py`
- `tests/llm/test_openai_compatible.py`

## 任务 1：更新依赖与配置契约

**文件：**

- 修改：`pyproject.toml`
- 修改：`.env`
- 修改：`src/config.py`
- 测试：`tests/llm/test_provider_config.py`

- [ ] **步骤 1：先写失败测试，覆盖新配置项默认值**

```python
def test_settings_contains_runtime_timeouts() -> None:
    settings = Settings()
    assert settings.log_api_timeout_seconds == 6
    assert settings.llm_final_timeout_seconds == 20
    assert settings.source_preference == "external_first"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/llm/test_provider_config.py -v`
预期：失败，提示 `Settings` 缺少新字段。

- [ ] **步骤 3：补齐依赖与配置实现（最小可用）**

```toml
# pyproject.toml
dependencies = [
  "langgraph>=0.2.0",
  "llama-index>=0.11.0",
  "llama-index-embeddings-openai>=0.2.0",
  "sqlalchemy>=2.0.0",
  "pymysql>=1.1.0",
]
```

```python
# config.py（示意）
log_api_timeout_seconds: int = Field(default=6, validation_alias=AliasChoices("LOG_API_TIMEOUT_SEC"))
rag_timeout_seconds: int = Field(default=8, validation_alias=AliasChoices("RAG_TIMEOUT_SEC"))
llm_final_timeout_seconds: int = Field(default=20, validation_alias=AliasChoices("LLM_FINAL_TIMEOUT_SEC"))
source_preference: str = Field(default="external_first", validation_alias=AliasChoices("SOURCE_PREFERENCE"))
mysql_dsn: str = Field(default="", validation_alias=AliasChoices("MYSQL_DSN"))
```

- [ ] **步骤 4：运行测试确认通过**

运行：`pytest tests/llm/test_provider_config.py -v`
预期：通过。

- [ ] **步骤 5：提交**

```bash
git add pyproject.toml .env src/config.py tests/llm/test_provider_config.py
git commit -m "feat: add langgraph/llamaindex runtime settings"
```

## 任务 2：定义领域枚举与结果模型（失败原因、恢复动作、置信等级）

**文件：**

- 修改：`src/domain/enums.py`
- 修改：`src/domain/models.py`
- 修改：`tests/domain/test_models.py`

- [ ] **步骤 1：新增失败枚举/置信等级的失败测试**

```python
def test_analysis_result_confidence_is_enum() -> None:
    result = AnalysisResult(root_cause="x", confidence="high")
    assert result.confidence == "high"


def test_failure_reason_enum_contains_llm_need_more_info() -> None:
    assert FailureReason.LLM_NEED_MORE_INFO == "llm_need_more_info"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/domain/test_models.py -v`
预期：失败，`confidence` 仍为 float。

- [ ] **步骤 3：实现枚举与模型升级**

```python
class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureReason(StrEnum):
    EXTERNAL_TIMEOUT = "external_timeout"
    EXTERNAL_5XX = "external_5xx"
    EXTERNAL_429 = "external_429"
    EXTERNAL_AUTH_ERROR = "external_auth_error"
    EXTERNAL_BAD_REQUEST = "external_bad_request"
    NO_DATA_FOUND = "no_data_found"
    EVIDENCE_CONFLICT = "evidence_conflict"
    LLM_TIMEOUT = "llm_timeout"
    LLM_NEED_MORE_INFO = "llm_need_more_info"
    LLM_INVALID_OUTPUT = "llm_invalid_output"
    UNKNOWN_ERROR = "unknown_error"


class AnalysisResult(BaseModel):
    root_cause: str
    confidence: ConfidenceLevel
```

- [ ] **步骤 4：运行测试确认通过**

运行：`pytest tests/domain/test_models.py -v`
预期：通过。

- [ ] **步骤 5：提交**

```bash
git add src/domain/enums.py src/domain/models.py tests/domain/test_models.py
git commit -m "feat: add failure/action/confidence enums"
```

## 任务 3：实现日志与代码检索 Provider（外部优先 + fallback）

**文件：**

- 新建：`src/providers/logs/base.py`
- 新建：`src/providers/logs/external_http.py`
- 新建：`src/providers/logs/local_case.py`
- 新建：`src/providers/code/base.py`
- 新建：`src/providers/code/git_remote.py`
- 新建：`src/providers/code/local_repo.py`
- 修改：`src/agent/skills/logs.py`
- 修改：`src/agent/skills/code.py`
- 测试：`tests/providers/test_log_providers.py`
- 测试：`tests/providers/test_code_providers.py`
- 测试：`tests/tools/test_mock_tools.py`

- [ ] **步骤 1：先写失败测试，覆盖重试与 fallback**

```python
def test_log_skill_fallbacks_to_local_when_external_timeout() -> None:
    external = FakeExternalProvider(raise_timeout=True)
    local = FakeLocalProvider(items=[{"message": "DBConnectionTimeout"}])
    skill = LogAnalysisSkill(external=external, local=local, max_retry=2)

    result = skill.execute(request, artifacts=[])

    assert result[0].content == "DBConnectionTimeout"
    assert external.calls == 3
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/providers/test_log_providers.py tests/providers/test_code_providers.py -v`
预期：失败，当前 skill 仅支持 mock tool。

- [ ] **步骤 3：实现 Provider 抽象与分类错误映射**

```python
@dataclass
class ProviderError(Exception):
    reason: FailureReason
    message: str


class LogProvider(Protocol):
    def query(self, request: AnalysisRequest) -> list[dict]:
        raise NotImplementedError
```

```python
class ExternalHttpLogProvider:
    def query(self, request: AnalysisRequest) -> list[dict]:
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=1, read=5, write=5, pool=1)) as client:
                response = client.get(
                    f"{self.base_url}/logs/search",
                    params={"service": request.service_name, "q": request.query, "time_range": request.time_range},
                    headers={"Authorization": f"Bearer {self.api_token}"},
                )
                response.raise_for_status()
                return response.json().get("items", [])
        except httpx.ReadTimeout as exc:
            raise ProviderError(FailureReason.EXTERNAL_TIMEOUT, str(exc))
```

- [ ] **步骤 4：改造 Skill 为“外部优先 + fallback + retry”**

```python
for attempt in range(max_retry + 1):
    try:
        items = self.external.query(request)
        break
    except ProviderError as err:
        if err.reason in RETRYABLE_REASONS and attempt < max_retry:
            continue
        items = self.local.query(request)
        break
```

- [ ] **步骤 5：运行测试确认通过**

运行：
- `pytest tests/providers/test_log_providers.py -v`
- `pytest tests/providers/test_code_providers.py -v`
- `pytest tests/tools/test_mock_tools.py -v`

预期：通过，覆盖重试 + fallback。

- [ ] **步骤 6：提交**

```bash
git add src/providers src/agent/skills/logs.py src/agent/skills/code.py tests/providers tests/tools/test_mock_tools.py
git commit -m "feat: add external-first providers with fallback"
```

## 任务 4：RAG 接入 LlamaIndex 检索器

**文件：**

- 新建：`src/rag/llamaindex_retriever.py`
- 修改：`src/agent/skills/knowledge.py`
- 修改：`src/main_support.py`
- 测试：`tests/rag/test_llamaindex_retriever.py`
- 修改：`tests/rag/test_keyword_retriever.py`

- [ ] **步骤 1：编写失败测试，验证 LlamaIndex provider 的文档召回**

```python
def test_llamaindex_retriever_returns_top_k(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("DBConnectionTimeout", encoding="utf-8")
    retriever = LlamaIndexRetriever(knowledge_dir=tmp_path, top_k=2)
    docs = retriever.search("下单 DBConnectionTimeout")
    assert docs
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/rag/test_llamaindex_retriever.py -v`
预期：失败，类不存在。

- [ ] **步骤 3：实现 LlamaIndex 检索器（先走本地索引）**

```python
class LlamaIndexRetriever:
    def __init__(self, knowledge_dir: Path, top_k: int = 3):
        self.documents = SimpleDirectoryReader(str(knowledge_dir)).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    def search(self, query: str) -> list[RetrievedDocument]:
        nodes = self.index.as_retriever(similarity_top_k=self.top_k).retrieve(query)
        return [RetrievedDocument(document_id=node.node_id, score=float(node.score or 0), content=node.text) for node in nodes]
```

- [ ] **步骤 4：替换 KnowledgeSearchSkill 的依赖注入**

```python
retriever = LlamaIndexRetriever(settings.knowledge_dir)
skill_registry["knowledge_search"] = KnowledgeSearchSkill(retriever)
```

- [ ] **步骤 5：运行测试确认通过**

运行：
- `pytest tests/rag/test_llamaindex_retriever.py -v`
- `pytest tests/rag/test_keyword_retriever.py -v`

预期：通过。

- [ ] **步骤 6：提交**

```bash
git add src/rag/llamaindex_retriever.py src/agent/skills/knowledge.py src/main_support.py tests/rag
git commit -m "feat: integrate llamaindex rag retriever"
```

## 任务 5：实现短期/长期记忆 + 会话经验向量化

**文件：**

- 新建：`src/memory/short_term.py`
- 新建：`src/memory/long_term_store.py`
- 新建：`src/memory/experience_store.py`
- 新建：`src/session/lifecycle.py`
- 修改：`src/api/routes/sessions.py`
- 修改：`src/api/routes/chat.py`
- 修改：`src/main_support.py`
- 测试：`tests/memory/test_short_term_memory.py`
- 测试：`tests/memory/test_long_term_store.py`
- 测试：`tests/memory/test_experience_store.py`
- 测试：`tests/api/test_sessions_lifecycle.py`

- [ ] **步骤 1：编写失败测试，覆盖短期压缩与结束归档**

```python
def test_short_term_memory_summarizes_after_k_turns() -> None:
    mem = ShortTermMemory(max_turns=2)
    mem.append("u1", "a1")
    mem.append("u2", "a2")
    mem.append("u3", "a3")
    assert mem.running_summary


def test_session_end_archives_mysql_and_vector_experiences() -> None:
    lifecycle = SessionLifecycle(long_term=FakeLongTerm(), experience=FakeExperienceStore())
    lifecycle.end_session(
        user_id="U123",
        session_id="S1",
        evidence=[{"source": "log", "text": "DBConnectionTimeout", "quality": "high"}],
        final={"root_cause": "数据库连接池耗尽", "confidence": "high"},
    )
    assert lifecycle.long_term.saved
    assert lifecycle.experience.saved_count >= 1


def test_session_start_loads_recent_long_term_summaries() -> None:
    lifecycle = SessionLifecycle(long_term=FakeLongTerm(rows=[{"conversation_summary": "历史故障摘要"}]))
    context = lifecycle.start_session(user_id="U123", session_id="S2")
    assert context.long_term_summaries
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/memory tests/api/test_sessions_lifecycle.py -v`
预期：失败，模块未实现。

- [ ] **步骤 3：实现短期记忆与摘要压缩**

```python
class ShortTermMemory:
    def append(self, user_text: str, assistant_text: str) -> None:
        self.turns.append({"user": user_text, "assistant": assistant_text})

    def maybe_compress(self, llm_client, model: str) -> bool:
        if len(self.turns) <= self.max_turns:
            return False
        self.running_summary = llm_client.generate_text(model=model, system_prompt="summarize", user_prompt=str(self.turns), fallback_answer="").answer
        self.turns = self.turns[-self.max_turns :]
        return True

    def build_prompt_context(self) -> dict:
        return {"recent_turns": self.turns, "running_summary": self.running_summary}
```

- [ ] **步骤 4：实现长期记忆 MySQL 与经验向量存储**

```python
class MySqlLongTermStore:
    def archive_session(
        self,
        *,
        user_id: str,
        session_id: str,
        conversation_value: dict,
        conversation_summary: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        self.repo.insert(
            user_id=user_id,
            session_id=session_id,
            conversation_value=conversation_value,
            conversation_summary=conversation_summary,
            started_at=started_at,
            ended_at=ended_at,
        )


class ExperienceVectorStore:
    def extract_experiences(self, evidence: list[dict], final_result: dict, max_items: int = 3) -> list[str]:
        lines = [f"根因：{final_result['root_cause']}"]
        lines.extend([f"证据：{item['source']}::{item['text']}" for item in evidence[: max_items - 1]])
        return lines[:max_items]

    def upsert_experiences(self, *, user_id: str, session_id: str, experiences: list[str]) -> None:
        self.vector_client.upsert(user_id=user_id, session_id=session_id, texts=experiences)
```

- [ ] **步骤 5：在 session 生命周期 API 中挂接**

```python
@router.post("/sessions/end")
def end_session(payload: EndSessionRequest, lifecycle=Depends(get_session_lifecycle)):
    lifecycle.end_session(
        user_id=payload.user_id,
        session_id=payload.session_id,
        evidence=payload.evidence,
        final=payload.final_result,
    )
    return {"status": "archived"}
```

```python
@router.post("/sessions/start")
def start_session(payload: StartSessionRequest, lifecycle=Depends(get_session_lifecycle)):
    context = lifecycle.start_session(user_id=payload.user_id, session_id=payload.session_id)
    return {"status": "started", "context": context.model_dump()}
```

- [ ] **步骤 6：运行测试确认通过**

运行：
- `pytest tests/memory -v`
- `pytest tests/api/test_sessions_lifecycle.py -v`

预期：通过。

- [ ] **步骤 7：提交**

```bash
git add src/memory src/session/lifecycle.py src/api/routes/sessions.py src/api/routes/chat.py src/main_support.py tests/memory tests/api/test_sessions_lifecycle.py
git commit -m "feat: add short/long memory and experience vectorization"
```

## 任务 6：引入 LangGraph 状态机与 `should_finalize` 规则

**文件：**

- 新建：`src/graph/state.py`
- 新建：`src/graph/nodes.py`
- 新建：`src/graph/workflow.py`
- 修改：`src/agent/planner.py`
- 修改：`src/agent/orchestrator.py`
- 修改：`src/main_support.py`
- 测试：`tests/graph/test_judge_next.py`
- 测试：`tests/graph/test_workflow.py`
- 修改：`tests/agent/test_planner.py`
- 修改：`tests/agent/test_orchestrator.py`

- [ ] **步骤 1：编写失败测试，锁定 finalize 布尔规则**

```python
def test_should_finalize_requires_cross_validated_and_no_conflict_and_covers_issue() -> None:
    state = make_state(
        evidence=[
            EvidenceItem(source="log", quality="high", text="DBConnectionTimeout"),
            EvidenceItem(source="knowledge", quality="high", text="数据库连接池耗尽"),
        ],
        parsed_keywords=["DBConnectionTimeout"],
        failure=None,
    )
    assert should_finalize(state) is True
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/graph/test_judge_next.py -v`
预期：失败，graph 模块未实现。

- [ ] **步骤 3：实现 AgentState 与节点函数**

```python
class AgentState(TypedDict, total=False):
    request: AnalysisRequest
    parsed: ParsedRequest
    plan: list[PlanStep]
    cursor: int
    evidence: list[EvidenceItem]
    failure: FailureState | None
    decision: str
```

```python
def should_finalize(agent_state: AgentState) -> bool:
    keywords = agent_state["parsed"].keywords or agent_state["parsed"].query_terms
    evidence_sources = {e.source for e in agent_state["evidence"] if e.quality == "high"}
    cross_validated = len(evidence_sources) >= 2
    no_conflict = not agent_state.get("failure") or agent_state["failure"].reason != FailureReason.EVIDENCE_CONFLICT
    covers_issue = any(keyword in e.text for e in agent_state["evidence"] for keyword in keywords)
    return cross_validated and no_conflict and covers_issue
```

- [ ] **步骤 4：组装 LangGraph 工作流并接入 Orchestrator**

```python
def build_analysis_graph(deps: GraphDependencies):
    graph = StateGraph(AgentState)
    graph.add_node("parse_request", parse_request)
    graph.add_node("plan_steps", plan_steps)
    graph.add_node("execute_step", execute_step)
    graph.add_node("judge_next", judge_next)
    graph.add_node("replan_if_needed", replan_if_needed)
    graph.add_node("final_synthesis", final_synthesis)
    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "plan_steps")
    graph.add_edge("plan_steps", "execute_step")
    graph.add_conditional_edges("judge_next", route_next_step)
    graph.add_edge("final_synthesis", END)
    return graph.compile()
```

- [ ] **步骤 5：运行测试确认通过**

运行：
- `pytest tests/graph/test_judge_next.py -v`
- `pytest tests/graph/test_workflow.py -v`
- `pytest tests/agent/test_planner.py tests/agent/test_orchestrator.py -v`

预期：通过。

- [ ] **步骤 6：提交**

```bash
git add src/graph src/agent/planner.py src/agent/orchestrator.py src/main_support.py tests/graph tests/agent/test_planner.py tests/agent/test_orchestrator.py
git commit -m "feat: implement langgraph dynamic workflow"
```

## 任务 7：改造 Final Synthesis（confidence 枚举 + Prompt 规则）

**文件：**

- 修改：`src/agent/skills/rca.py`
- 修改：`src/llm/prompts.py`
- 修改：`src/llm/openai_compatible.py`
- 修改：`tests/llm/test_openai_compatible.py`
- 修改：`tests/api/test_analyze.py`

- [ ] **步骤 1：写失败测试，验证 confidence 枚举解析**

```python
def test_parse_json_content_accepts_confidence_enum() -> None:
    content = '{"root_cause":"x","confidence":"high","suggestions":[],"summary":"ok"}'
    payload = _parse_json_content(content)
    assert payload["confidence"] == "high"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/llm/test_openai_compatible.py tests/api/test_analyze.py -v`
预期：失败，当前解析器仅支持 float。

- [ ] **步骤 3：更新 Prompt 契约与回退逻辑**

```python
CONFIDENCE_RULES = """
high: 至少两个独立工具来源强证据
medium: 单一可靠来源强证据或多来源弱证据
low: 缺乏直接证据，主要依赖推理
"""
```

```python
def _extract_confidence_field(content: str) -> str | None:
    match = re.search(r'"confidence"\\s*:\\s*"(high|medium|low)"', content)
    return match.group(1) if match else None
```

- [ ] **步骤 4：运行测试确认通过**

运行：
- `pytest tests/llm/test_openai_compatible.py -v`
- `pytest tests/api/test_analyze.py -v`

预期：通过。

- [ ] **步骤 5：提交**

```bash
git add src/agent/skills/rca.py src/llm/prompts.py src/llm/openai_compatible.py tests/llm/test_openai_compatible.py tests/api/test_analyze.py
git commit -m "feat: switch confidence to qualitative enum"
```

## 任务 8：API 与端到端收口（analyze/chat/sessions）

**文件：**

- 修改：`src/api/schemas.py`
- 修改：`src/api/routes/analyze.py`
- 修改：`src/api/routes/chat.py`
- 修改：`src/api/routes/sessions.py`
- 修改：`src/main.py`
- 测试：`tests/api/test_analyze.py`
- 测试：`tests/api/test_chat.py`
- 测试：`tests/api/test_sessions_lifecycle.py`

- [ ] **步骤 1：新增/改造 API 测试（先失败）**

```python
def test_analyze_returns_confidence_enum() -> None:
    body = client.post(
        "/api/v1/analyze",
        json={"query": "下单失败，日志里有 DBConnectionTimeout", "service_name": "order-service", "environment": "prod"},
    ).json()
    assert body["result"]["confidence"] in {"high", "medium", "low"}


def test_session_end_archives_and_returns_status() -> None:
    body = client.post("/api/v1/sessions/end", json={"user_id": "U1", "session_id": "S1"}).json()
    assert body["status"] == "archived"


def test_session_start_returns_history_context() -> None:
    body = client.post("/api/v1/sessions/start", json={"user_id": "U1", "session_id": "S2"}).json()
    assert body["status"] == "started"
    assert "context" in body
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest tests/api/test_analyze.py tests/api/test_chat.py tests/api/test_sessions_lifecycle.py -v`
预期：失败，响应模型尚未更新。

- [ ] **步骤 3：实现 API 契约升级**

```python
class AnalyzeResponse(BaseModel):
    session_id: str
    status: str
    result: dict  # result.confidence -> high|medium|low
    trace_summary: dict
    evaluation: dict


class EndSessionRequest(BaseModel):
    user_id: str
    session_id: str
```

- [ ] **步骤 4：运行 API 测试确认通过**

运行：`pytest tests/api/test_analyze.py tests/api/test_chat.py tests/api/test_sessions_lifecycle.py -v`
预期：通过。

- [ ] **步骤 5：提交**

```bash
git add src/api src/main.py tests/api
git commit -m "feat: finalize api contracts for langgraph aiops assistant"
```

## 任务 9：全量验证与发布前检查

**文件：**

- 修改：`README.md`（如需补充运行说明）
- 修改：`docs/AGENT_DESIGN.md`（同步 LangGraph + Memory 新设计）

- [ ] **步骤 1：运行分层测试套件**

运行：
- `pytest tests/domain tests/providers tests/rag tests/graph tests/memory -v`
- `pytest tests/agent tests/api tests/llm tests/tools tests/evaluation -v`

预期：全部通过。

- [ ] **步骤 2：运行一次端到端手工验证**

运行：
- `uvicorn ai_ops_agent.main:app --reload`
- `curl -X POST http://127.0.0.1:8000/api/v1/analyze -H 'Content-Type: application/json' -d '{"query":"下单失败，TraceID:abc-123","service_name":"order-service"}'`

预期：返回结构化 RCA，`confidence` 为 `high|medium|low`。

- [ ] **步骤 3：更新文档并复跑关键测试**

运行：`pytest tests/api/test_analyze.py tests/graph/test_judge_next.py -v`
预期：通过。

- [ ] **步骤 4：最终提交**

```bash
git add README.md docs/AGENT_DESIGN.md
git commit -m "docs: update architecture and runbook for langgraph assistant"
```

## 执行顺序建议

1. 先完成任务 1-2（契约先行）。
2. 再完成任务 3-4（数据获取能力）。
3. 然后完成任务 5-6（状态机与记忆）。
4. 最后完成任务 7-9（对外契约收口与全量验证）。

## 风险检查清单

- [ ] 外部日志接口超时后确实重试再 fallback
- [ ] `should_finalize` 未依赖浮点阈值
- [ ] `confidence` 只输出 `high|medium|low`
- [ ] 会话结束时同时写 MySQL 与向量经验库
- [ ] 新会话可读取用户历史摘要/经验增强规划
