# Plan-ReAct-RePlan Skill Layer 实施计划（V3）

> **给智能代理工作者：** 必需：使用 qsuperpowers:subagent-driven-development（如果有子代理可用）或 qsuperpowers:executing-plans 来执行此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 将当前 Agent 主链路替换为 `Query Rewrite + Knowledge Retrieve + Planner(Hypothesis Generation) + ReAct Executor + Evidence-based Evaluator + RePlan Loop + Root Cause Analysis`，并落地 L1/L2 Skill 分层。

**架构：** 先做 Query Rewrite 与分库知识检索（Domain/Case/Code），再由 Planner 基于问题与知识生成 `hypothesis + investigation_goals`。ReAct Executor 负责“怎么查”，按 objective 与当前证据动态选 Skill，并维护带 hypothesis 节点的 Evidence Graph。Evaluator 只输出 `supported | insufficient | unsupported`。`unsupported` 时 RePlan 只记录失败原因与被证伪假设，然后回到 Planner 重生假设，避免职责重叠。

**技术栈：** Python 3.14, LangGraph, FastAPI, pytest, 现有 LLM/Prompt/Tool 体系

---

## 目标链路（必须实现）

```text
用户问题
↓
Intent
↓
Query Rewrite
↓
Knowledge Retrieve（Dynamic）
↓
Planner（Hypothesis Generation）
↓
ReAct Executor（Evidence Graph）
↓
Evaluator（Evidence-based）
↓
Root Cause

        ↑
        │
      RePlan
        │
        └───── Planner
```

```text
控制闭环（概念层）：
Intent -> Planner -> ReAct Executor -> Evaluator -> RePlan -> Planner
```

```text
Evaluator 状态机：
supported   -> Root Cause
insufficient -> Knowledge Retrieve（补证后继续执行当前 hypothesis）
unsupported -> RePlan -> Planner（新 hypothesis）
```

---

## 文件结构规划

### 新增文件

- `src/flow/modules/agent_executor_graph/graph/query_rewrite/query_rewrite.py`
- `src/flow/modules/agent_executor_graph/graph/knowledge_retrieve/knowledge_retrieve.py`
- `src/flow/modules/agent_executor_graph/graph/evaluator/evaluator.py`
- `src/flow/modules/agent_executor_graph/graph/replan/replan.py`
- `src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py`
- `src/flow/modules/agent_executor_graph/skills/l1_business.py`
- `src/flow/modules/agent_executor_graph/skills/l2_generic.py`
- `src/flow/modules/agent_executor_graph/skills/router.py`
- `src/tests/flow/test_query_rewrite.py`
- `src/tests/flow/test_knowledge_retrieve.py`
- `src/tests/flow/test_evaluator.py`
- `src/tests/flow/test_replan.py`
- `src/tests/flow/test_root_cause.py`
- `src/tests/flow/test_skill_router.py`
- `src/tests/flow/test_evidence_graph.py`

### 修改文件

- `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
- `src/flow/modules/agent_executor_graph/agent_state.py`
- `src/flow/modules/agent_executor_graph/plan_step.py`
- `src/flow/modules/agent_executor_graph/graph/planner/planner.py`
- `src/flow/modules/agent_executor_graph/graph/executor/executor.py`
- `src/flow/modules/agent_executor_graph/graph/intent_decide/intent_decide.py`
- `src/tests/flow/test_agent_executor_graph.py`
- `src/tests/flow/test_planner.py`
- `src/tests/flow/test_plan_react_nodes.py`

### 退出主路由（可保留文件）

- `src/flow/modules/agent_executor_graph/graph/observer/observer.py`
- `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`
- `src/flow/modules/agent_executor_graph/graph/result_validate/result_validate.py`
- `src/flow/modules/agent_executor_graph/graph/retry_router/retry_router.py`

---

### 任务 1: 重定义状态契约（Hypothesis + Investigation Goals + RePlan Memory + Evidence Graph）

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/agent_state.py`
- 修改: `src/flow/modules/agent_executor_graph/plan_step.py`
- 测试: `src/tests/flow/test_state_build.py`

- [ ] **步骤 1: 编写失败测试（新状态字段）**

```python
def test_state_supports_hypothesis_investigation_goals_and_evidence_graph() -> None:
    state = {
        "plan": {
            "hypothesis": "MQ异常",
            "investigation_goals": ["确认MQ发送状态", "确认MQ消费状态"],
        },
        "rejected_hypothesis": ["数据库异常"],
        "replan_reason": "证据与假设冲突",
        "execution": {
            "evidence_graph": {
                "hypothesis": "MQ异常",
                "evidence": [],
                "supported": None
            }
        },
    }
    assert state["plan"]["hypothesis"] == "MQ异常"
    assert state["plan"]["investigation_goals"][0] == "确认MQ发送状态"
    assert state["rejected_hypothesis"][0] == "数据库异常"
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_state_build.py -q`
预期: 失败，字段未定义或未传递

- [ ] **步骤 3: 实现最小状态定义**

```python
class AgentState(TypedDict, total=False):
    plan: Dict[str, Any]  # hypothesis + investigation_goals
    execution: Dict[str, Any]  # observations + evidence_graph
    evaluation: Dict[str, Any]
    rejected_hypothesis: List[str]
    replan_reason: str
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_state_build.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/agent_state.py src/flow/modules/agent_executor_graph/plan_step.py src/tests/flow/test_state_build.py
git commit -m "refactor: add hypothesis-goals-evidence-graph state contract"
```

---

### 任务 2: Query Rewrite 节点落地

**文件：**

- 创建: `src/flow/modules/agent_executor_graph/graph/query_rewrite/query_rewrite.py`
- 测试: `src/tests/flow/test_query_rewrite.py`

- [ ] **步骤 1: 编写失败测试（rewrite 结果）**

```python
def test_query_rewrite_extracts_ids_keywords_and_time_window() -> None:
    out = run({"question": "traceId=ops_slugger_xxx 订单失败"})
    assert out["query_rewrite"]["trace_id"]
    assert out["query_rewrite"]["keywords"]
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_query_rewrite.py -q`
预期: 失败，模块不存在

- [ ] **步骤 3: 实现最小 rewrite 逻辑**

```python
state["query_rewrite"] = {
    "normalized_query": question,
    "trace_id": _extract_trace_id(question),
    "order_id": _extract_order_id(question),
    "keywords": _extract_keywords(question),
    "time_window": _infer_time_window(question),
}
state["route"] = "knowledge_retrieve"
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_query_rewrite.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/query_rewrite/query_rewrite.py src/tests/flow/test_query_rewrite.py
git commit -m "feat: add query rewrite node"
```

---

### 任务 3: Planner 改为输出 hypothesis + investigation_goals

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/graph/planner/planner.py`
- 修改: `src/llm/prompts/planner_system_prompt.txt`
- 修改: `src/llm/prompts/planner_user_prompt.txt`
- 测试: `src/tests/flow/test_planner.py`

- [ ] **步骤 1: 编写失败测试（新输出结构）**

```python
def test_planner_outputs_hypothesis_and_investigation_goals(monkeypatch):
    monkeypatch.setattr(..., lambda **_: '{"hypothesis":"MQ异常","investigation_goals":["确认MQ发送状态","确认MQ消费状态"]}')
    state = planner_run({"question": "MQ积压", "intent_type": "OPS_ANALYSIS"})
    assert state["plan"]["hypothesis"] == "MQ异常"
    assert state["plan"]["investigation_goals"][0] == "确认MQ发送状态"


def test_planner_avoids_rejected_hypothesis(monkeypatch):
    monkeypatch.setattr(..., lambda **_: '{"hypothesis":"支付异常","investigation_goals":["确认支付回调状态"]}')
    state = planner_run({
        "question": "订单失败",
        "rejected_hypothesis": ["MQ异常", "支付异常"],
        "replan_reason": "支付链路证据不足",
    })
    assert state["plan"]["hypothesis"] not in {"MQ异常", "支付异常"}
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_planner.py -q`
预期: 失败，尚未写入新结构

- [ ] **步骤 3: 实现 planner 输出映射**

```python
state["plan"] = {
    "hypothesis": str(parsed.get("hypothesis") or ""),
    "investigation_goals": _normalize_goals(parsed.get("investigation_goals")),
}
# planner prompt 需显式注入被证伪假设与失败原因
prompt_context["rejected_hypothesis"] = state.get("rejected_hypothesis", [])
prompt_context["replan_reason"] = state.get("replan_reason", "")
state["route"] = "executor"
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_planner.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/planner/planner.py src/llm/prompts/planner_*.txt src/tests/flow/test_planner.py
git commit -m "refactor: planner outputs hypothesis and investigation goals"
```

---

### 任务 4: Dynamic Knowledge Retrieval（分库：Domain/Case/Code，支持预规划检索与补证检索）

**文件：**

- 创建: `src/flow/modules/agent_executor_graph/graph/knowledge_retrieve/knowledge_retrieve.py`
- 测试: `src/tests/flow/test_knowledge_retrieve.py`

- [ ] **步骤 1: 编写失败测试（动态检索输入）**

```python
def test_knowledge_retrieve_supports_pre_plan_retrieve() -> None:
    state = run({
        "query_rewrite": {"normalized_query": "订单失败", "keywords": ["生单失败"]},
    })
    assert "domain_docs" in state["knowledge_context"]
    assert "case_docs" in state["knowledge_context"]
    assert "code_docs" in state["knowledge_context"]


def test_knowledge_retrieve_uses_hypothesis_and_current_objective() -> None:
    state = run({
        "plan": {"hypothesis": "MQ异常", "investigation_goals": ["确认MQ发送状态"]},
        "execution": {"goal_index": 0},
    })
    assert state["knowledge_context"]["query_basis"]["hypothesis"] == "MQ异常"
    assert "domain_docs" in state["knowledge_context"]
    assert "case_docs" in state["knowledge_context"]
    assert "code_docs" in state["knowledge_context"]
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_knowledge_retrieve.py -q`
预期: 失败，模块不存在

- [ ] **步骤 3: 实现动态检索逻辑**

```python
basis = {
    "hypothesis": plan_hypothesis,  # 预规划阶段可为空
    "objective": current_objective,  # 预规划阶段可为空
    "query_rewrite": query_rewrite,
}
state["knowledge_context"] = {
    "query_basis": basis,
    "domain_docs": retrieve_domain_docs(basis),
    "case_docs": retrieve_case_docs(basis),
    "code_docs": retrieve_code_docs(basis),
}
state["route"] = "planner" if not plan_hypothesis else "executor"
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_knowledge_retrieve.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/knowledge_retrieve/knowledge_retrieve.py src/tests/flow/test_knowledge_retrieve.py
git commit -m "feat: add dynamic knowledge retrieval node"
```

---

### 任务 5: Skill Router 改签名并接入 ReAct Executor

**文件：**

- 创建: `src/flow/modules/agent_executor_graph/skills/l1_business.py`
- 创建: `src/flow/modules/agent_executor_graph/skills/l2_generic.py`
- 创建: `src/flow/modules/agent_executor_graph/skills/router.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`
- 测试: `src/tests/flow/test_skill_router.py`
- 测试: `src/tests/flow/test_plan_react_nodes.py`

- [ ] **步骤 1: 编写失败测试（新签名）**

```python
def test_skill_router_uses_hypothesis_evidence_objective_signature() -> None:
    out = route_skill(
        hypothesis="MQ异常",
        current_evidence={"signals": ["send_timeout"]},
        objective="确认MQ发送状态",
    )
    assert out["skill"] in {"getMqSendStatus", "queryLog"}
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_skill_router.py -q`
预期: 失败，签名不匹配

- [ ] **步骤 3: 实现 Skill Router 与 L1/L2 路由**

```python
def route_skill(hypothesis: str, current_evidence: dict[str, Any], objective: str) -> dict[str, Any]:
    ...
```

- [ ] **步骤 4: ReAct Executor 改为调用新签名并保留兼容日志**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_plan_react_nodes.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/skills src/flow/modules/agent_executor_graph/graph/executor/executor.py src/tests/flow/test_skill_router.py src/tests/flow/test_plan_react_nodes.py
git commit -m "refactor: update skill router signature and executor integration"
```

---

### 任务 6: Executor 增加 Hypothesis-first Evidence Graph 持久化

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`
- 测试: `src/tests/flow/test_evidence_graph.py`

- [ ] **步骤 1: 编写失败测试（Evidence Graph 写入）**

```python
def test_executor_persists_hypothesis_centric_evidence_graph() -> None:
    out = executor_run({...})
    graph = out["execution"]["evidence_graph"]
    assert graph["hypothesis"] == "MQ异常"
    assert isinstance(graph["evidence"], list)
    assert graph["supported"] in {True, False, None}
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_evidence_graph.py -q`
预期: 失败，字段不存在

- [ ] **步骤 3: 实现最小 Evidence Graph 模型**

```python
execution.setdefault("evidence_graph", {
    "hypothesis": current_hypothesis,
    "evidence": [],
    "supported": None,
})
execution["evidence_graph"]["evidence"].append(evidence_item)
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_evidence_graph.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/executor/executor.py src/tests/flow/test_evidence_graph.py
git commit -m "feat: persist evidence graph in executor"
```

---

### 任务 7: Evaluator + RePlan（RePlan 必须回到 Planner）

**文件：**

- 创建: `src/flow/modules/agent_executor_graph/graph/evaluator/evaluator.py`
- 创建: `src/flow/modules/agent_executor_graph/graph/replan/replan.py`
- 测试: `src/tests/flow/test_evaluator.py`
- 测试: `src/tests/flow/test_replan.py`

- [ ] **步骤 1: 编写失败测试（Evidence-based decision）**

```python
def test_evaluator_decides_replan_when_hypothesis_not_supported() -> None:
    out = evaluator_run({...})
    assert out["evaluation"]["status"] == "unsupported"
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_evaluator.py tests/flow/test_replan.py -q`
预期: 失败，模块不存在

- [ ] **步骤 3: 实现 Evaluator/RePlan 核心逻辑**

```python
if evidence_supports_hypothesis:
    status = "supported"
elif evidence_conflicts_hypothesis:
    status = "unsupported"
else:
    status = "insufficient"
```

```python
# replan.py
state["replan_reason"] = reason
state.setdefault("rejected_hypothesis", []).append(current_hypothesis)
state["route"] = "planner"
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_evaluator.py tests/flow/test_replan.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/evaluator/evaluator.py src/flow/modules/agent_executor_graph/graph/replan/replan.py src/tests/flow/test_evaluator.py src/tests/flow/test_replan.py
git commit -m "feat: add evidence-based evaluator and replan-to-planner loop"
```

---

### 任务 8: Root Cause Analysis 节点

**文件：**

- 创建: `src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py`
- 测试: `src/tests/flow/test_root_cause.py`

- [ ] **步骤 1: 编写失败测试（输出根因、证据链、过程）**

```python
def test_root_cause_outputs_report_with_evidence_graph_refs() -> None:
    out = root_cause_run({...})
    assert out["analysis"]["root_cause"]
    assert out["analysis"]["evidence_chain"]
    assert out["analysis"]["troubleshooting_process"]
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_root_cause.py -q`
预期: 失败，模块不存在

- [ ] **步骤 3: 实现最小 root cause 汇总**

```python
state["analysis"] = {
    "root_cause": root_cause,
    "evidence_chain": chain,
    "troubleshooting_process": process,
    "reply": final_reply,
}
state["status"] = "finished"
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_root_cause.py -q`
预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py src/tests/flow/test_root_cause.py
git commit -m "feat: add root cause analysis node"
```

---

### 任务 9: 重排主图为新顺序并完成回归

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/intent_decide/intent_decide.py`
- 测试: `src/tests/flow/test_agent_executor_graph.py`

- [ ] **步骤 1: 编写失败测试（新主链路）**

```python
def test_ops_analysis_uses_new_main_chain_order(): ...
def test_replan_routes_back_to_planner(): ...
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py -q`
预期: 失败，旧链路仍在

- [ ] **步骤 3: 实现主图连边**

```python
graph.add_edge("intent_decide", "query_rewrite")
graph.add_edge("query_rewrite", "knowledge_retrieve")
graph.add_conditional_edges("knowledge_retrieve", retrieve_route, {
    "planner": "planner",    # 初次检索，供 Planner 生成 hypothesis
    "executor": "executor",  # insufficient 补证后，继续当前 hypothesis 执行
})
graph.add_edge("planner", "executor")
graph.add_edge("executor", "evaluator")
graph.add_conditional_edges("evaluator", route, {
    "supported": "root_cause",
    "insufficient": "knowledge_retrieve",
    "unsupported": "replan",
})
graph.add_edge("replan", "planner")
```

- [ ] **步骤 4: 运行关键回归**

运行:

```bash
cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py tests/flow/test_planner.py tests/flow/test_plan_react_nodes.py tests/flow/test_evidence_graph.py -q
```

预期: 通过

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/build_langgraph_graph.py src/flow/modules/agent_executor_graph/graph/intent_decide/intent_decide.py src/tests/flow/test_agent_executor_graph.py
git commit -m "refactor: switch main chain to knowledge-first hypothesis-driven react loop"
```

---

## 执行约束

- 严格 TDD：先失败测试，再最小实现，再回归。
- 每个任务独立提交，避免跨任务混改。
- Skill Router 必须使用以下签名：

```python
route_skill(
    hypothesis,
    current_evidence,
    objective,
)
```

- Planner 输出结构固定为：

```json
{
  "hypothesis": "MQ异常",
  "investigation_goals": [
    "确认MQ发送状态",
    "确认MQ消费状态"
  ]
}
```

- Planner Prompt 必须注入被证伪假设：

```text
以下假设已经被证伪，请不要重复生成：
- MQ异常
- 支付异常
```

---

## 里程碑验收

- M1: `query_rewrite -> knowledge_retrieve -> planner` 路径打通
- M2: `planner(hypothesis/investigation_goals) + route_skill(new signature)` 可用
- M3: `executor hypothesis-first evidence_graph + evaluator(status)` 可用
- M4: 主图切换为新链路，`replan -> planner` 回环生效
- M5: `root_cause` 输出根因、证据链、排查过程并通过 flow 回归
