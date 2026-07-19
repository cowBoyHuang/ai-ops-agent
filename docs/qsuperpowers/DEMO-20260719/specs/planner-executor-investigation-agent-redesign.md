# Planner Executor Investigation Agent Redesign

## Background

The current troubleshooting graph already has a hypothesis-driven loop, but the responsibilities are still mixed:

- Planner generates `hypothesis + investigation_goals`, then downstream code still decides how to progress goals.
- Reactor owns goal-loop control, retry policy, queryLog fallback, required-field extraction, and executor invocation.
- Executor owns tool selection, prompt schema construction, tool dispatch, and evidence writing.
- Observer owns goal status, replan decisions, required-field validation, and final routing.

This makes the runtime hard to reason about. Planner, Reactor, Executor, and Observer all participate in investigation strategy. The result is closer to a fixed workflow with corrective loops than a clean investigation agent.

The previous tool migration already unified executable capabilities behind `@tool` and `tool.registry`. This redesign builds on that foundation.

## Goal

Rebuild the main troubleshooting path into a complete-plan investigation agent:

```text
Context Builder
  -> Planner
  -> Plan Controller
  -> Capability Router
  -> Domain Executor
  -> Plan Controller
  -> Summary
```

Planner runs once at the start and produces a complete investigation plan. Planner runs again only when a replan trigger fires.

## Non-Goals

- Do not return to per-round Planner decision making.
- Do not let Planner call tools or choose concrete tool parameters.
- Do not let Capability Router revise goals or infer investigation strategy.
- Do not merge all tools back into a single global Executor prompt.
- Do not replace the `@tool` registry introduced by the unified tool annotation work.
- Do not redesign external log, RAG, or Code Index service protocols.

## Design

### Top-Level Flow

The main troubleshooting path becomes:

```text
state_build
  -> intent_decide
  -> query_rewrite
  -> knowledge_retrieve
  -> planner
  -> plan_controller
  -> capability_router
  -> log_executor | code_executor | knowledge_executor | config_executor
  -> plan_controller
  -> summary
  -> finish
```

`fixed_flow_execute` can remain for business-consult short-circuit paths if the intent router still needs that branch. The main runtime troubleshooting path no longer uses the old `reactor -> observer -> replan` loop as the core architecture.

### Responsibilities

#### Planner

Planner is responsible for:

- Understanding user query, context, RAG context, and prior evidence during replan.
- Producing one complete investigation plan.
- Assigning each goal a required capability.
- Defining success criteria, expected evidence, finish criteria, and replan triggers.

Planner is not responsible for:

- Calling tools.
- Selecting executor implementations.
- Choosing log app names, log files, query terms, method names, or Code Index parameters.
- Advancing current goal during normal execution.

#### Plan Controller

Plan Controller is responsible for:

- Initializing investigation runtime state after Planner.
- Selecting the next pending goal whose dependencies are satisfied.
- Marking goals as `pending`, `running`, `succeeded`, `failed`, or `skipped`.
- Deciding whether to retry the current goal, continue to the next goal, finish, replan, or fallback.
- Enforcing retry and replan budgets.

Plan Controller does not call tools and does not rewrite Executor evidence.

#### Capability Router

Capability Router is a deterministic mapping layer:

```python
CAPABILITY_MAP = {
    "runtime_evidence": "LogExecutor",
    "code_analysis": "CodeExecutor",
    "business_validation": "KnowledgeExecutor",
    "config_analysis": "ConfigExecutor",
}
```

Router responsibilities:

- Validate `goal.required_capability`.
- Return the executor name and that executor's allowed tool names.
- Return a normalized unsupported result for unsupported capability.

Router does not decide investigation strategy and does not mutate the goal.

Supported route result:

```json
{
  "result_id": "route_g1_1",
  "goal_id": "g1",
  "required_capability": "runtime_evidence",
  "executor": "LogExecutor",
  "allowed_tools": ["queryLog", "dependency_log_query", "getFlightCreateOrderResult", "getCreateOrderResult"],
  "ok": true,
  "error": ""
}
```

Supported route results are not goal results. Plan Controller must not update goal completion status from a supported route result. A supported route result only selects the executor; Plan Controller consumes it and sets `pending_execution`.

Unsupported route result reuses the normalized executor-result envelope so Plan Controller can consume it through the same error path:

```json
{
  "executor": "",
  "result_id": "route_g1_1",
  "goal_id": "g1",
  "goal_complete": false,
  "status": "unsupported",
  "summary": "",
  "facts": {},
  "evidence": [],
  "artifacts": [],
  "confidence": 0.0,
  "error": "capability_not_supported"
}
```

#### Domain Executors

Each Executor owns one investigation domain and internally performs ReAct over its allowed tools.

`LogExecutor`

- Purpose: collect runtime evidence, identify failure stage, extract error codes, failure reasons, and call-chain clues.
- Allowed tools: `queryLog`, `dependency_log_query`, `getFlightCreateOrderResult`, `getCreateOrderResult`.

`CodeExecutor`

- Purpose: locate relevant code paths, methods, classes, and call relationships.
- Allowed tools: `searchMethod`, `locateCode`, `analyzeCodeFromLogs`, `analyzeCodeForBusinessConsult`.

`KnowledgeExecutor`

- Purpose: validate business rules, runbooks, FAQ, historical case evidence, and domain explanations.
- Allowed tools: `rag_parent_chunk_query`, `knowledge_lookup`.

`ConfigExecutor`

- Purpose: reserved for configuration analysis.
- First version returns `capability_not_supported` when no reliable config tools exist. Router still treats `config_analysis` as a known capability and routes it to `ConfigExecutor`.

Executors return normalized evidence/results. Executors do not write `investigation.evidence` and do not advance plan state directly.

### Investigation Plan Schema

Planner outputs `InvestigationPlanV2`:

```json
{
  "plan_id": "plan_001",
  "hypothesis": "订单失败可能发生在生单链路的业务校验阶段",
  "goals": [
    {
      "id": "g1",
      "goal": "定位异常发生阶段",
      "required_capability": "runtime_evidence",
      "priority": 1,
      "required": true,
      "success_criteria": ["明确失败服务", "明确错误码或失败日志"],
      "expected_evidence": ["log_event", "error_code"],
      "depends_on": []
    },
    {
      "id": "g2",
      "goal": "验证失败是否符合业务规则",
      "required_capability": "business_validation",
      "priority": 2,
      "required": true,
      "success_criteria": ["找到业务规则或历史 case"],
      "expected_evidence": ["business_doc"],
      "depends_on": ["g1"]
    }
  ],
  "finish_criteria": ["根因明确", "证据链闭合", "用户问题已被直接回答"],
  "replan_triggers": ["goal_unexecutable", "evidence_conflict", "missing_required_context", "capability_not_supported"]
}
```

Rules:

- `goals` must be ordered by intended execution sequence using `priority`.
- Each goal must have exactly one `required_capability`.
- Each goal must include `required: true|false`. Missing `required` is treated as `true`.
- A goal may depend on earlier goal IDs.
- Planner must not include tool names, app codes, log names, query strings, line numbers, or tool parameters.
- `finish_criteria` describes when the final answer is allowed.
- `replan_triggers` describes when normal plan execution should stop and Planner should regenerate a full plan.

### State Design

Add a dedicated `investigation` runtime object to `AgentState` while keeping existing fields during migration:

```python
investigation = {
    "plan": InvestigationPlanV2,
    "current_goal_id": "g1",
    "goal_status": {
        "g1": "pending|running|succeeded|failed|skipped"
    },
    "evidence": [
        {
            "goal_id": "g1",
            "capability": "runtime_evidence",
            "executor": "LogExecutor",
            "summary": "...",
            "facts": {},
            "evidence": [
                {
                    "type": "log_event",
                    "source": "getCreateOrderResult",
                    "content": "...",
                    "confidence": 0.9
                }
            ],
            "artifacts": [],
            "confidence": 0.8
        }
    ],
    "events": [
        {
            "type": "planner|router|executor|controller|replan",
            "message": "...",
            "payload": {}
        }
    ],
    "pending_execution": {
        "goal_id": "g1",
        "executor": "LogExecutor",
        "attempt": 1
    },
    "last_route_result": {},
    "last_executor_result": {},
    "consumed_result_ids": [],
    "retry_counts_by_goal": {
        "g1": 0
    },
    "max_retries_per_goal": 2,
    "replan_count": 0,
    "max_replans": 1,
    "failure_reason": ""
}
```

Ownership rules:

- Planner writes `investigation.plan`.
- Plan Controller writes `current_goal_id`, `goal_status`, route decisions, and controller events.
- Capability Router writes router events and selected executor metadata.
- Executors return normalized evidence/results and executor events to Plan Controller.
- Plan Controller is the single owner of persisting normalized executor evidence into `investigation.evidence`.
- Router writes `last_route_result` plus router events. Router never writes `pending_execution`.
- Executor writes `last_executor_result` with a unique `result_id`.
- Plan Controller appends consumed `result_id` values to `consumed_result_ids`.
- Supported route results are consumed by appending the route `result_id` to `consumed_result_ids`, setting `pending_execution`, and clearing `last_route_result`; they do not change goal status.
- Unsupported route results are consumed like failed normalized goal results and may change goal status.
- Executor results are consumed by persisting evidence, updating goal status, clearing `pending_execution`, and clearing `last_executor_result`.
- First-version defaults are `max_retries_per_goal=2` and `max_replans=1`.
- Summary reads `investigation.plan`, `investigation.evidence`, and `finish_criteria`.
- Replan preserves the old plan and failure reason in `events`, then replaces `investigation.plan`.

### Executor Output Schema

All domain Executor graph nodes return a normalized result to Plan Controller:

```json
{
  "executor": "LogExecutor",
  "result_id": "exec_g1_1",
  "goal_id": "g1",
  "goal_complete": true,
  "status": "succeeded",
  "summary": "定位到总单生单返回失败，错误码为 10321",
  "facts": {
    "errorCode": "10321",
    "failedService": "f_tts_trade_order"
  },
  "evidence": [
    {
      "type": "log_event",
      "source": "getCreateOrderResult",
      "content": "...",
      "confidence": 0.9
    }
  ],
  "artifacts": [],
  "confidence": 0.9,
  "error": ""
}
```

Failure results use the same envelope:

```json
{
  "executor": "LogExecutor",
  "result_id": "exec_g1_1",
  "goal_id": "g1",
  "goal_complete": false,
  "status": "failed",
  "summary": "",
  "facts": {},
  "evidence": [],
  "artifacts": [],
  "confidence": 0.0,
  "error": "missing_required_context"
}
```

Plan Controller persists an executor result into `investigation.evidence` using this transform:

```python
{
    "goal_id": result["goal_id"],
    "capability": current_goal["required_capability"],
    "executor": result["executor"],
    "summary": result["summary"],
    "facts": result["facts"],
    "evidence": result["evidence"],
    "artifacts": result["artifacts"],
    "confidence": result["confidence"],
    "status": result["status"],
    "error": result["error"],
}
```

The nested `evidence` array is retained so Summary can cite concrete sources by goal and executor.

Allowed `status` values:

- `succeeded`: success criteria are satisfied.
- `failed`: the goal is not complete and Controller must decide retry, replan, skip, or fallback.
- `unsupported`: router or executor cannot support the requested capability.

Allowed `error` values for non-success results:

- `tool_error`: a tool failed or returned an execution error.
- `empty_evidence`: tool execution succeeded but produced no useful evidence.
- `missing_required_context`: trace ID, order ID, time window, or another required context is missing.
- `goal_unexecutable`: executor cannot turn the goal into a valid action.
- `evidence_conflict`: new evidence contradicts existing evidence or the plan hypothesis.
- `capability_not_supported`: router or executor does not support the requested capability.
- `invalid_executor_result`: executor output failed schema normalization.

### Prompt Updates

#### Planner Prompts

Update `planner_system_prompt.txt` and `planner_user_prompt.txt`:

- Output `InvestigationPlanV2`, not `hypothesis + investigation_goals`.
- Require every goal to contain `id`, `goal`, `required_capability`, `priority`, `required`, `success_criteria`, `expected_evidence`, and `depends_on`.
- Require `finish_criteria` and `replan_triggers`.
- Explicitly forbid tool names and concrete tool parameters.
- For replan, include previous plan, current evidence, failed goal, and failure reason.

#### Executor Prompts

Replace the single global executor prompt with domain-specific prompts, or parameterize one shared template by domain:

- `log_executor_react_system_prompt.txt`
- `code_executor_react_system_prompt.txt`
- `knowledge_executor_react_system_prompt.txt`
- `config_executor_react_system_prompt.txt` if needed.

Executor prompt input:

- `current_goal`
- `success_criteria`
- `expected_evidence`
- `allowed_tools`
- `tool_schemas_json` filtered to allowed tools
- `existing_evidence`
- `structured_context`
- `question`

Executor prompt output:

```json
{
  "thought": "why this action serves current goal",
  "action": {
    "tool_name": "registeredToolName",
    "params": {}
  },
  "final_evidence": {
    "summary": "",
    "facts": {},
    "evidence": [],
    "confidence": 0.0
  },
  "goal_complete": false
}
```

This is the internal LLM ReAct step schema, not the graph-node return shape. The domain Executor wrapper must:

1. Parse each LLM ReAct step.
2. Invoke only allowed registered tools.
3. Decide whether the goal is complete against `success_criteria`.
4. Convert `final_evidence + goal_complete` into the normalized Executor Output Schema.
5. Return only the normalized Executor Output Schema to Plan Controller.

Executors can make multiple ReAct steps internally, but the outer graph sees one normalized executor result for the goal.

#### Controller Prompt

First version should prefer deterministic rules. If an LLM is needed later, add `plan_controller_system_prompt.txt` and `plan_controller_user_prompt.txt`.

Controller prompt, if introduced, may only classify:

- `goal_succeeded`
- `retry_goal`
- `next_goal`
- `replan`
- `finish`
- `fallback`

It must not call tools.

#### Summary Prompt

Update summary prompts to read:

- User question.
- Investigation plan.
- Goal statuses.
- Evidence list.
- Finish criteria.

Summary must cite evidence by goal and executor. It should explicitly state uncertainty when finish criteria are not fully satisfied but fallback answer is required.

### Control Flow

Plan Controller behavior:

1. If no `investigation.plan`, route to Planner.
2. If the previous node returned a supported route result for `current_goal_id`, consume it by setting `pending_execution`, recording a controller event, and routing to the selected executor.
3. If the previous node returned an unsupported route result or executor result for `current_goal_id`, consume that goal result first:
   - append normalized evidence when present;
   - update `goal_status[current_goal_id]` to `succeeded`, `failed`, `skipped`, or keep `running` only when a retry is about to be scheduled;
   - record a controller event;
   - clear any pending execution marker so the same result is not consumed twice.
4. If all required goals are `succeeded`, route to Summary. In the first version, `finish_criteria` is used by Summary for answer quality, not as a semantic controller gate.
5. If the current goal succeeded, select the next pending goal whose dependencies are satisfied.
6. If the current goal failed with retryable error and retry budget remains, retry the same goal through Router.
7. If the current goal failed with a replan error, route to Replan only when the error is listed in `plan.replan_triggers` and `replan_count < max_replans`; otherwise route to Fallback.
8. If the current goal failed with an optional unsupported capability, mark it `skipped` and continue.
9. If no pending goal has satisfied dependencies and not all required goals have succeeded, mark blocked dependent goals as `failed` when required and `skipped` when optional, then route to Replan if `replan_count < max_replans`; otherwise route to Fallback.
10. If replan budget is exhausted, route to Fallback.

Error routing table:

| Error | Default action |
| --- | --- |
| `tool_error` | Retry same goal while retry budget remains; otherwise replan if `replan_count < max_replans`; otherwise fallback |
| `empty_evidence` | Retry same goal while retry budget remains; otherwise replan if `replan_count < max_replans`; otherwise fallback |
| `missing_required_context` | Replan if `replan_count < max_replans`; otherwise fallback |
| `goal_unexecutable` | Replan if `replan_count < max_replans`; otherwise fallback |
| `evidence_conflict` | Replan if `replan_count < max_replans`; otherwise fallback |
| `capability_not_supported` | Replan for required goals if `replan_count < max_replans`; fallback for required goals if budget is exhausted; skip optional goals |
| `invalid_executor_result` | Retry once; otherwise fallback |

Retry and replan budget rules:

- Retry is tracked in `retry_counts_by_goal[goal_id]`.
- A retry is allowed while `retry_counts_by_goal[goal_id] < max_retries_per_goal`.
- First-version `max_retries_per_goal` defaults to `2`.
- Replan is allowed while `replan_count < max_replans`.
- First-version `max_replans` defaults to `1`.
- Increment `investigation.replan_count` exactly once when a replan attempt is accepted before calling Planner.
- After a replan, retry counts are reset for goals in the new plan.

`plan.replan_triggers` is authoritative for replan classification. The parser must default missing or empty `replan_triggers` to:

```json
["goal_unexecutable", "evidence_conflict", "missing_required_context", "capability_not_supported"]
```

Errors not listed in `plan.replan_triggers` cannot route to Replan and must follow retry, skip, or fallback behavior.

### Replan

Replan is not normal per-round planning. It is an exceptional path.

Replan input:

- Original user query.
- Current context.
- Previous plan.
- Goal statuses.
- Evidence gathered so far.
- Failed goal and failure reason.
- Rejected or contradicted assumptions.

Replan output:

- A new complete `InvestigationPlanV2`.
- The new plan should either avoid failed assumptions or explicitly add a goal to resolve the missing context.

When a new plan is accepted, Plan Controller must:

- replace `investigation.plan`;
- initialize `goal_status` for the new plan;
- set `current_goal_id` to the first executable goal;
- clear `pending_execution`, `last_route_result`, and `last_executor_result`;
- reset `retry_counts_by_goal` for goals in the new plan;
- preserve old plans and failure reasons in `events`.

### Error Handling

`capability_not_supported`

- Router returns this only for unknown capability.
- First-version `ConfigExecutor` returns this when it has no reliable config tools.
- Controller triggers replan if the goal is required.
- Controller may skip the goal only when `goal.required == false`.

`missing_required_context`

- Executor returns this when it cannot proceed without trace ID, order ID, time window, or another required context.
- Controller triggers replan or fallback depending on budget and whether the missing context can be inferred.

`tool_error`

- Executor handles local tool retry.
- Repeated tool errors produce normalized result `status: "failed"` with `error: "tool_error"`.

`empty_evidence`

- Executor marks goal incomplete unless success criteria permit empty result.
- Controller retries or replans.

`evidence_conflict`

- Controller records conflict and triggers replan.

### Testing

Add or update focused tests:

- Planner parses and normalizes `InvestigationPlanV2`.
- Planner fallback returns a complete plan with valid capabilities.
- Planner prompts do not request tool names or tool parameters.
- Router maps each supported capability to the expected executor.
- Router returns structured `capability_not_supported` for unknown capability.
- Plan Controller initializes plan runtime.
- Plan Controller advances goals by dependency and priority.
- Plan Controller routes to retry, replan, summary, and fallback under the correct conditions.
- Each Executor receives only allowed tool schemas.
- Each Executor returns normalized evidence schema.
- Summary reads `investigation.evidence` rather than old scattered fields.
- Integration test covers `runtime_evidence -> business_validation -> summary`.

### Migration Strategy

1. Add `InvestigationPlanV2` and investigation runtime state.
2. Add Planner parser and prompt changes.
3. Add Plan Controller node and tests.
4. Add Capability Router node and tests.
5. Split domain Executors around filtered `@tool` schemas.
6. Wire new graph path.
7. Update summary input and prompts.
8. Retire old `reactor`, `observer`, and global executor from the main path after replacement tests pass.

## Acceptance Criteria

- Planner runs once per normal investigation and only reruns through explicit replan.
- Planner outputs a complete `InvestigationPlanV2`.
- Planner output contains capabilities, success criteria, expected evidence, finish criteria, and replan triggers.
- Planner output does not contain concrete tool calls or tool parameters.
- Capability Router is deterministic and side-effect-light.
- Each domain Executor sees only its allowed `@tool` schemas.
- Executors return normalized evidence/results and do not persist investigation state or advance plan state directly.
- Plan Controller is the single owner of writing executor evidence into `investigation.evidence`.
- Plan Controller is the only component that advances goals, retries goals, finishes, or triggers replan.
- Summary uses `investigation.evidence` and finish criteria.
- The old `reactor -> observer -> replan` loop is no longer the main troubleshooting path.
- Focused tests pass for Planner, Router, Controller, Executors, Summary integration, and prompt constraints.
