# Reactor-Centric Hypothesis Loop 实施计划

> **给智能代理工作者：** 必需：使用 qsuperpowers:subagent-driven-development（如果有子代理可用）或 qsuperpowers:executing-plans 来执行此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 将现有 `executor->evaluator` 流程改造为 `reactor(内循环)->observer(外层路由)`，保证计划完整执行后再总结，并按环境变量控制 replan 预算。

**架构：** Planner 仅产出 hypothesis/goals；Reactor 在单 goal 内维护 `act_times/retry` 并输出 `reactor_report`；Observer 读取 report 决策 next/replan/finish，并在 finish 前把“用户问题+全目标执行报告”喂给 LLM 统一总结。外部日志打印改为长度统计，禁止正文落盘。

**技术栈：** Python 3.14、LangGraph、Pytest、内置 logging、现有 `llm.chat_with_llm`。

---

## 范围检查

本计划只覆盖一个子系统：`agent_executor_graph` 执行闭环 + `log.py` 打印策略，不拆分子项目。

## 文件结构与职责

- `src/flow/modules/agent_executor_graph/agent_state.py`
  - 增加 `reactor_runtime / goal_reports / replan_context / final_summary_input` 等状态字段。
- `src/flow/modules/agent_executor_graph/graph/state_build/state_build.py`
  - 注入默认预算：`AIOPS_MAX_REPLAN`（默认 0）、`max_act_times`（默认 5）。
- `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
  - 主图改为 `... -> planner -> reactor -> observer -> ...`。
- `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`
  - 重写为单-goal内循环执行器，输出标准 `reactor_report`。
- `src/flow/modules/agent_executor_graph/graph/observer/observer.py`
  - 重写为唯一外层路由器，负责 next/replan/finish 和 replan_context 组装。
- `src/flow/modules/agent_executor_graph/graph/replan/replan.py`
  - 仅维护 replan 元信息，不直接写新 plan。
- `src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py`
  - 调整为消费 `goal_reports` 的证据汇总（或并入 finish）。
- `src/log/log.py`
  - `query_external_logs` 仅打印长度统计与样本长度，不打印正文。
- `src/tests/flow/test_state_build.py`
  - 新增环境变量预算默认值测试。
- `src/tests/flow/test_agent_executor_graph.py`
  - 改主图测试：覆盖 reactor/observer 路由。
- `src/tests/flow/test_replan.py`
  - 校验 observer 触发 replan 的上下文字段。
- 新增 `src/tests/flow/test_reactor.py`
  - 单 goal act_times/retry/goal_status 覆盖。
- 新增 `src/tests/flow/test_observer.py`
  - next_goal/replan/finish 三分支覆盖。
- 新增 `src/tests/log_query/test_query_external_logs_logging.py`
  - 校验日志不含正文 content。

---

### 任务 1: 状态与预算初始化改造

**文件：**
- 修改: `src/flow/modules/agent_executor_graph/agent_state.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/state_build/state_build.py`
- 测试: `src/tests/flow/test_state_build.py`

- [ ] **步骤 1: 编写失败测试（预算默认值与新状态字段）**

```python
# src/tests/flow/test_state_build.py
from __future__ import annotations

import os

from flow.modules.agent_executor_graph.graph.state_build.state_build import run as state_build_run


def test_state_build_defaults_replan_budget_and_reactor_budget(monkeypatch) -> None:
    monkeypatch.delenv("AIOPS_MAX_REPLAN", raising=False)
    result = state_build_run({"context": {"question": "订单失败"}})
    assert result.get("max_replan") == 0
    execution = dict(result.get("execution") or {})
    assert execution.get("max_act_times") == 5
    assert execution.get("goal_reports") == []
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_state_build.py::test_state_build_defaults_replan_budget_and_reactor_budget -q`
预期: 失败（字段未注入或默认值不对）。

- [ ] **步骤 3: 编写最小实现**

```python
# src/flow/modules/agent_executor_graph/graph/state_build/state_build.py
import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


state["max_replan"] = raw_context.get("max_replan", state.get("max_replan", _env_int("AIOPS_MAX_REPLAN", 0)))

execution = raw_context.get("execution", state.get("execution", {}))
if isinstance(execution, dict):
    execution.setdefault("max_act_times", 5)
    execution.setdefault("goal_reports", [])
    execution.setdefault("reactor_runtime", {})
state["execution"] = execution
```

- [ ] **步骤 4: 更新类型定义**

```python
# src/flow/modules/agent_executor_graph/agent_state.py
class AgentState(TypedDict, total=False):
    ...
    replan_context: Dict[str, Any]
    final_summary_input: Dict[str, Any]
    execution: Dict[str, Any]  # 内含 goal_reports/reactor_runtime/max_act_times
```

- [ ] **步骤 5: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_state_build.py -q`
预期: 通过。

- [ ] **步骤 6: 提交**

```bash
git add src/flow/modules/agent_executor_graph/agent_state.py src/flow/modules/agent_executor_graph/graph/state_build/state_build.py src/tests/flow/test_state_build.py
git commit -m "feat: initialize reactor/replan runtime budgets from state and env"
```

---

### 任务 2: 主图切换到 Reactor + Observer

**文件：**
- 修改: `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
- 测试: `src/tests/flow/test_agent_executor_graph.py`

- [ ] **步骤 1: 编写失败测试（主图不再走 evaluator）**

```python
# src/tests/flow/test_agent_executor_graph.py

def test_graph_routes_executor_phase_via_reactor_and_observer(monkeypatch):
    import flow.modules.agent_executor_graph.build_langgraph_graph as graph_builder

    monkeypatch.setattr(graph_builder, "reactor_run", lambda payload: {**dict(payload), "route": "observer"})
    monkeypatch.setattr(graph_builder, "observer_run", lambda payload: {**dict(payload), "route": "finish", "status": "finished", "response": {"chatId": "c", "status": "finished", "message": "ok"}})
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py -q`
预期: 失败（当前图仍绑定 evaluator）。

- [ ] **步骤 3: 修改主图节点与路由**

```python
# src/flow/modules/agent_executor_graph/build_langgraph_graph.py
from flow.modules.agent_executor_graph.graph.reactor.reactor import run as reactor_run
from flow.modules.agent_executor_graph.graph.observer.observer import run as observer_run

graph.add_node("reactor", reactor_run)
graph.add_node("observer", observer_run)
...
graph.add_edge("planner", "reactor")
graph.add_edge("reactor", "observer")
# observer 内部 route 决策: reactor/replan/finish/fallback
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py -q`
预期: 通过。

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/build_langgraph_graph.py src/tests/flow/test_agent_executor_graph.py
git commit -m "refactor: route hypothesis loop through reactor and observer"
```

---

### 任务 3: Reactor 单 goal 内循环执行

**文件：**
- 修改: `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`
- 可复用: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`（提取工具执行 helper）
- 测试: `src/tests/flow/test_reactor.py`（新建）

- [ ] **步骤 1: 编写失败测试（actTimes/retry/goal_status）**

```python
# src/tests/flow/test_reactor.py
from __future__ import annotations

from flow.modules.agent_executor_graph.graph.reactor.reactor import run as reactor_run


def test_reactor_marks_goal_failed_after_two_retries(monkeypatch):
    monkeypatch.setattr(
        "flow.modules.agent_executor_graph.graph.reactor.reactor._execute_action",
        lambda **_: {"ok": False, "error": "timeout", "evidence": []},
    )
    out = reactor_run(
        {
            "plan": {"investigation_goals": ["确认MQ发送状态"]},
            "execution": {"goal_index": 0, "max_act_times": 5, "reactor_runtime": {}},
        }
    )
    report = dict(out.get("current_step_result") or {}).get("reactor_report") or {}
    assert report.get("goal_status") == "failed"
    assert int(report.get("retry_count") or 0) == 2
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_reactor.py -q`
预期: 失败（当前 reactor 实现非该协议）。

- [ ] **步骤 3: 实现 reactor_report 协议与内循环**

```python
# src/flow/modules/agent_executor_graph/graph/reactor/reactor.py
while act_times < max_act_times:
    act_times += 1
    action = decide_action_with_llm(...)
    raw = _execute_action(action=action, state=state)
    action_chain.append({...})
    if raw.get("ok"):
        if goal_objective_met(...):
            goal_status = "success"
            break
    else:
        retry_count += 1
        if retry_count >= 2:
            goal_status = "failed"
            failure_reason = str(raw.get("error") or "retry_exhausted")
            break

if act_times >= max_act_times and goal_status == "in_progress":
    goal_status = "unexecutable"
    failure_reason = "act_times_exhausted"
```

- [ ] **步骤 4: 输出标准结构给 observer**

```python
state["current_step_result"] = {
    "reactor_report": {
        "goal_index": goal_index,
        "goal_objective": objective,
        "goal_status": goal_status,
        "act_times": act_times,
        "max_act_times": max_act_times,
        "retry_count": retry_count,
        "max_retry": 2,
        "action_chain": action_chain,
        "goal_conclusion": conclusion,
        "plan_unexecutable": goal_status == "unexecutable",
        "failure_reason": failure_reason,
    }
}
state["route"] = "observer"
```

- [ ] **步骤 5: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_reactor.py -q`
预期: 通过。

- [ ] **步骤 6: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/reactor/reactor.py src/tests/flow/test_reactor.py
# 如抽取了 helper，同步 add executor.py
# git add src/flow/modules/agent_executor_graph/graph/executor/executor.py
git commit -m "feat: implement per-goal reactor loop with actTimes and retry budget"
```

---

### 任务 4: Observer 路由与 RePlan 上下文

**文件：**
- 修改: `src/flow/modules/agent_executor_graph/graph/observer/observer.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/replan/replan.py`
- 测试: `src/tests/flow/test_observer.py`（新建）
- 测试: `src/tests/flow/test_replan.py`

- [ ] **步骤 1: 编写失败测试（失败时 observer 构造 replan_context）**

```python
# src/tests/flow/test_observer.py

def test_observer_builds_replan_context_when_goal_failed() -> None:
    from flow.modules.agent_executor_graph.graph.observer.observer import run as observer_run

    out = observer_run(
        {
            "plan": {"hypothesis": "MQ异常", "investigation_goals": ["确认MQ发送状态"]},
            "execution": {"goal_index": 0, "goal_reports": []},
            "max_replan": 1,
            "replan_count": 0,
            "current_step_result": {
                "reactor_report": {
                    "goal_index": 0,
                    "goal_objective": "确认MQ发送状态",
                    "goal_status": "failed",
                    "action_chain": [{"tool_name": "log_query", "result_summary": "timeout"}],
                    "failure_reason": "timeout",
                }
            },
        }
    )

    assert out.get("route") == "replan"
    assert dict(out.get("replan_context") or {}).get("failed_goal") == "确认MQ发送状态"
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_observer.py tests/flow/test_replan.py -q`
预期: 失败。

- [ ] **步骤 3: 实现 observer 三分支路由**

```python
# src/flow/modules/agent_executor_graph/graph/observer/observer.py
report = dict(dict(state.get("current_step_result") or {}).get("reactor_report") or {})
status = str(report.get("goal_status") or "")

if status == "success":
    append_goal_report(...)
    if has_next_goal(...):
        reset_reactor_runtime_for_next_goal(...)
        state["route"] = "reactor"
    else:
        state["route"] = "finish"  # finish 内再调用 LLM 做全局是否replan判定
elif status in {"failed", "unexecutable"}:
    state["replan_context"] = build_replan_context(...)
    if int(state.get("replan_count") or 0) < int(state.get("max_replan") or 0):
        state["route"] = "replan"
    else:
        state["route"] = "finish"
else:
    state["route"] = "fallback"
```

- [ ] **步骤 4: 让 replan 节点仅维护元信息**

```python
# src/flow/modules/agent_executor_graph/graph/replan/replan.py
state["replan_reason"] = reason_from_replan_context
state["rejected_hypothesis"] = append_current_hypothesis(...)
state["replan_count"] = int(state.get("replan_count") or 0) + 1
state["route"] = "planner"
```

- [ ] **步骤 5: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_observer.py tests/flow/test_replan.py -q`
预期: 通过。

- [ ] **步骤 6: 提交**

```bash
git add src/flow/modules/agent_executor_graph/graph/observer/observer.py src/flow/modules/agent_executor_graph/graph/replan/replan.py src/tests/flow/test_observer.py src/tests/flow/test_replan.py
git commit -m "feat: observer-driven replan routing with detailed replan context"
```

---

### 任务 5: Final Summarize（必须合并原始问题）

**文件：**
- 修改: `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py`
- 测试: `src/tests/flow/test_agent_executor_graph.py`

- [ ] **步骤 1: 编写失败测试（finish 使用 goal_reports 汇总）**

```python
# src/tests/flow/test_agent_executor_graph.py

def test_finish_node_merges_question_and_goal_reports_summary() -> None:
    from flow.modules.agent_executor_graph.build_langgraph_graph import _finish_node

    state = _finish_node(
        {
            "question": "为什么生单失败",
            "analysis": {},
            "execution": {
                "goal_reports": [
                    {
                        "goal_objective": "确认失败模块",
                        "action_chain": [{"tool_name": "log_query", "result_summary": "errorCode=404"}],
                        "goal_status": "success",
                    }
                ]
            },
        }
    )
    msg = str(dict(state.get("response") or {}).get("message") or "")
    assert "用户问题：为什么生单失败" in msg
    assert "errorCode=404" in msg
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py::test_finish_node_merges_question_and_goal_reports_summary -q`
预期: 失败。

- [ ] **步骤 3: 实现 finish 总结输入构造与LLM汇总**

```python
# src/flow/modules/agent_executor_graph/build_langgraph_graph.py
summary_input = {
    "user_question": question,
    "hypothesis": dict(state.get("plan") or {}).get("hypothesis"),
    "goals": list(dict(state.get("plan") or {}).get("investigation_goals") or []),
    "goal_reports": list(dict(state.get("execution") or {}).get("goal_reports") or []),
    "replan_count": int(state.get("replan_count") or 0),
    "max_replan": int(state.get("max_replan") or 0),
}
# 调 llm 生成结果摘要，拼接到“用户问题：...\n执行结论：...”
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow/test_agent_executor_graph.py -q`
预期: 通过。

- [ ] **步骤 5: 提交**

```bash
git add src/flow/modules/agent_executor_graph/build_langgraph_graph.py src/flow/modules/agent_executor_graph/graph/root_cause/root_cause.py src/tests/flow/test_agent_executor_graph.py
git commit -m "feat: summarize full goal reports with original user question in finish"
```

---

### 任务 6: 外部日志打印改为长度统计

**文件：**
- 修改: `src/log/log.py`
- 测试: `src/tests/log_query/test_query_external_logs_logging.py`（新建）

- [ ] **步骤 1: 编写失败测试（禁止 content 全文输出）**

```python
# src/tests/log_query/test_query_external_logs_logging.py
from __future__ import annotations

import logging

from log.log import EsResult, query_external_logs


def test_query_external_logs_logs_only_lengths(monkeypatch, caplog):
    def _fake_search_logs(**kwargs):
        return [EsResult(score=1.0, content="A" * 100), EsResult(score=2.0, content="B" * 50)]

    monkeypatch.setattr("log.log.search_logs", _fake_search_logs)

    with caplog.at_level(logging.INFO):
        query_external_logs(
            app_code="f_tts_trade_order",
            logname="ttsorder",
            begin_time="2026-06-14 17:20:42",
            end_time="2026-06-14 19:20:42",
            content={"match_phrase_list": ["trace"], "match_list": ["失败"]},
        )

    text = "\n".join(item.getMessage() for item in caplog.records)
    assert "content=" not in text
    assert "content_total_chars=" in text
    assert "max_item_chars=" in text
```

- [ ] **步骤 2: 运行测试验证失败**

运行: `cd src && ./.venv/bin/python -m pytest tests/log_query/test_query_external_logs_logging.py -q`
预期: 失败（当前仍打印 content 全文）。

- [ ] **步骤 3: 修改 query_external_logs 打印逻辑**

```python
# src/log/log.py
_LOGGER.info("log.query_external_logs.end app_code=%s hit_count=%d", final_app_code, len(rows))

total_chars = 0
max_chars = 0
for row in rows:
    content_len = len(str(getattr(row, "content", "") or ""))
    total_chars += content_len
    if content_len > max_chars:
        max_chars = content_len

avg_chars = (total_chars / len(rows)) if rows else 0
_LOGGER.info(
    "log.query_external_logs.summary app_code=%s hit_count=%d content_total_chars=%d max_item_chars=%d avg_item_chars=%.2f",
    final_app_code,
    len(rows),
    total_chars,
    max_chars,
    avg_chars,
)
for idx, row in enumerate(rows[:3], start=1):
    _LOGGER.info(
        "log.query_external_logs.sample idx=%d score=%.4f content_length=%d",
        idx,
        float(getattr(row, "score", 0.0) or 0.0),
        len(str(getattr(row, "content", "") or "")),
    )
```

- [ ] **步骤 4: 运行测试验证通过**

运行: `cd src && ./.venv/bin/python -m pytest tests/log_query/test_query_external_logs_logging.py -q`
预期: 通过。

- [ ] **步骤 5: 提交**

```bash
git add src/log/log.py src/tests/log_query/test_query_external_logs_logging.py
git commit -m "refactor: log external query metrics without dumping full content"
```

---

### 任务 7: 回归与端到端验证

**文件：**
- 测试: `src/tests/flow/test_reactor.py`
- 测试: `src/tests/flow/test_observer.py`
- 测试: `src/tests/flow/test_agent_executor_graph.py`
- 测试: `src/tests/flow/test_state_build.py`
- 测试: `src/tests/log_query/test_query_external_logs_logging.py`

- [ ] **步骤 1: 运行核心单测集**

运行:

```bash
cd src && ./.venv/bin/python -m pytest \
  tests/flow/test_state_build.py \
  tests/flow/test_reactor.py \
  tests/flow/test_observer.py \
  tests/flow/test_replan.py \
  tests/flow/test_agent_executor_graph.py \
  tests/log_query/test_query_external_logs_logging.py -q
```

预期: 全部通过。

- [ ] **步骤 2: 运行 flow 回归**

运行: `cd src && ./.venv/bin/python -m pytest tests/flow -q`
预期: 通过。

- [ ] **步骤 3: 提交回归修复（如有）**

```bash
git add src/tests/flow src/tests/log_query src/flow/modules/agent_executor_graph src/log/log.py
git commit -m "test: align flow regression with reactor-observer loop"
```

---

## 人工审查记录（替代 plan-document-reviewer 子代理）

由于当前会话未启用子代理派发，本计划采用人工自审，检查点：

- 设计一致性：与 `docs/qsuperpowers/AIOPS-20260614/specs/hypothesis-driven-reactor-loop-design.md` 一致。
- 职责边界：Planner/Reactor/Observer/Replan 职责不重叠。
- 测试闭环：每个任务都包含 fail -> pass -> commit。
- 非功能约束：外部日志只打印长度统计。

## 执行提示

- 执行技能：`@qsuperpowers:executing-plans`
- 若可用子代理：`@qsuperpowers:subagent-driven-development`
- 严格按任务顺序执行，每任务完成后提交。
