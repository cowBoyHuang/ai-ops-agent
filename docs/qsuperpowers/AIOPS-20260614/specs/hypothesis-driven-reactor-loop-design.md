# Hypothesis Driven Agent: Reactor-Centric Execution Loop Design

## 1. 目标与范围

本设计将当前链路改造为 **Planner + Reactor + Observer + RePlan** 的职责清晰架构，满足以下业务约束：

- 即使中途已定位到可能根因，也必须执行完整套 `investigation_goals` 后再统一总结。
- `reactor` 仅为当前 goal 服务，内部可执行工具调用、分析、重试。
- step 失败或 plan 不可执行时，`observer` 负责决定是否 replan。
- replan 上限由环境变量控制，默认 `0`。
- 最终回答必须合并：`用户原始问题 + 全部工具调用关键结果 + 结论`。
- 外部日志查询结果在本地日志中仅打印长度统计，不打印完整 `content` 全文。

不在本次范围：

- 新增外部存储或消息队列。
- RAG 数据源重构。

## 2. 目标链路

主链路改造为：

`User -> Intent -> Query Rewrite -> Planner -> Knowledge Retrieve -> Reactor(goal内循环) -> Observer -> (Next Goal | RePlan | Final Summarize)`

规则：

1. Planner 仅输出 `hypothesis + investigation_goals`。
2. Reactor 管理当前 goal 的动作链与重试。
3. Observer 是唯一 replan 决策点。
4. 所有 goals 完成后，调用 LLM 做全局判断（`finish/replan`）。
5. 若需 replan 但 `replan_count >= max_replan`，直接 `finish` 并说明未继续重规划原因。

## 3. 状态模型

### 3.1 plan

```json
{
  "hypothesis": "...",
  "investigation_goals": ["...", "..."]
}
```

### 3.2 execution（外层）

```json
{
  "goal_index": 0,
  "replan_count": 0,
  "max_replan": 0,
  "goal_reports": [],
  "final_decision": ""
}
```

- `max_replan` 来源：环境变量（默认 0）。
- `goal_reports`：每个 goal 的聚合执行报告（用于最终总结）。

### 3.3 reactor_runtime（内层，当前 goal）

```json
{
  "act_times": 0,
  "max_act_times": 5,
  "retry_count": 0,
  "max_retry": 2,
  "action_chain": [],
  "goal_status": "in_progress"
}
```

规则：

- `act_times` 按单个 goal 计数，切换 goal 重置为 0。
- 失败重试最多 2 次。

### 3.4 replan_context（observer 产出）

标准粒度（用户确认）：

```json
{
  "failed_goal": "...",
  "failure_reason": "...",
  "action_chain_summary": [
    {
      "tool": "...",
      "params_summary": {},
      "observation": "...",
      "result": "..."
    }
  ],
  "tool_failures": ["..."],
  "previous_hypothesis": "...",
  "rejected_hypothesis": ["..."]
}
```

## 4. Reactor 与 Observer 协议

### 4.1 reactor_report

Reactor 每次完成当前 goal 输出：

```json
{
  "goal_index": 0,
  "goal_objective": "确认MQ发送状态",
  "goal_status": "success",
  "act_times": 4,
  "max_act_times": 5,
  "retry_count": 1,
  "max_retry": 2,
  "action_chain": [],
  "goal_conclusion": "当前目标已得到支持证据",
  "plan_unexecutable": false,
  "failure_reason": ""
}
```

`goal_status` 允许值：`success | failed | unexecutable`。

### 4.2 observer 路由规则

1. `goal_status=success` 且存在下一个 goal：
- `goal_index += 1`
- 重置 `reactor_runtime`
- 路由回 `reactor`

2. `goal_status in (failed, unexecutable)`：
- 生成 `replan_context`
- 若 `replan_count < max_replan`，路由 `replan`
- 否则路由 `finish`

3. 所有 goals 完成：
- 将原始问题 + 全量 goal_reports 喂给 LLM
- LLM 输出 `finish/replan`
- 若输出 `replan` 但预算不足，仍 `finish`

## 5. 最终总结策略

最终总结输入：

- `user_question`
- `hypothesis`
- `investigation_goals`
- `goal_reports`（标准粒度）
- `replan_attempts`
- `stop_reason`

输出要求：

1. 必须显式包含“用户问题”。
2. 必须包含关键证据路径（按 goal 聚合）。
3. 若未 replan（例如 `max_replan=0`），需写明原因。

## 6. 环境变量

新增：

- `AIOPS_MAX_REPLAN`：外层 replan 最大次数，默认 `0`。

保留默认：

- Reactor `max_act_times=5`
- Reactor `max_retry=2`

## 7. 代码改造边界

### 7.1 必改文件

- `src/flow/modules/agent_executor_graph/build_langgraph_graph.py`
- `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`
- `src/flow/modules/agent_executor_graph/graph/observer/observer.py`
- `src/flow/modules/agent_executor_graph/graph/replan/replan.py`
- `src/flow/modules/agent_executor_graph/graph/planner/planner.py`
- `src/flow/modules/agent_executor_graph/graph/state_build/state_build.py`
- `src/flow/modules/agent_executor_graph/agent_state.py`
- `src/log/log.py`

### 7.2 可能删改

- `graph/evaluator/evaluator.py`：若职责被 observer 吸收，可降级为兼容层或移除路由入口。

## 8. 日志与可观测性

新增标准日志事件：

- `reactor.goal.start`
- `reactor.action.dispatch`
- `reactor.action.result`
- `reactor.goal.end`
- `observer.route.decide`
- `observer.replan.context`
- `final.summary.input`
- `final.summary.output`

目标：单条 request.log 可还原 goal 执行链、失败路径和 replan 决策依据。

### 8.1 外部日志打印收敛策略

针对 `query_external_logs` 增加日志收敛规范：

1. 禁止打印每条命中的完整 `content`。
2. 打印聚合指标：`hit_count`、`content_total_chars`、`max_item_chars`、`avg_item_chars`。
3. 如需排障定位，仅打印有限样本的摘要信息（例如前 3 条的 `idx + score + content_length`），不输出正文。

示例：

```text
log.query_external_logs.summary app_code=f_tts_trade_order hit_count=1000 content_total_chars=2843912 max_item_chars=47670 avg_item_chars=2843
log.query_external_logs.sample idx=1 score=273.0442 content_length=4210
log.query_external_logs.sample idx=2 score=272.9588 content_length=3897
```

## 9. 测试方案

### 9.1 单测

- Reactor：
  - 单 goal act_times 递增与上限终止
  - 失败重试 2 次
  - `success/failed/unexecutable` 输出正确
- Observer：
  - `next_goal/replan/finish` 路由
  - `max_replan=0` 时不进入 replan
- Log：
  - `query_external_logs` 不输出正文 `content`
  - 输出长度统计字段与 sample 长度字段

### 9.2 集成测试

- 场景1：中途已命中根因但仍执行完整 plan 后 finish。
- 场景2：step 失败触发 observer replan，且下一轮 planner 收到失败包。
- 场景3：all goals complete 后 LLM 判定 replan，但预算不足，最终 finish。

## 10. 风险与缓解

风险：

- goal 数多时时延升高。
- action_chain 过长导致总结 prompt 过大。

缓解：

- 总结阶段使用标准粒度压缩，不透传原始全文。
- 为每个 goal 限制 `act_times=5`，避免内循环失控。

## 11. 验收标准

1. 不再出现“中途定位到根因但提前结束/提前replan”。
2. 每个 goal 都有可追踪执行报告。
3. replan 仅由 observer 触发。
4. `AIOPS_MAX_REPLAN=0` 时，需 replan 的场景也能稳定 finish。
5. 最终输出包含“用户问题 + 执行证据 + 结论/限制”。
6. `request.log` 中不再出现外部日志全文，仅保留长度统计与样本长度信息。
