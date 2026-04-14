# AI Agent 运维助手设计文档（L1，LangGraph 动态重规划）

日期：2026-04-03  
状态：Draft（已与需求方对齐）

## 1. 背景与目标

本项目目标是构建一个 **L1 诊断型 AI Agent 运维助手**，用于故障分析与排障建议生成，不执行自动修复动作。

用户确认的核心链路：

`Plan（规划） -> 循环执行（MCP/Skills/RAG） -> LLM 最终生成`

本期能力范围：

- 日志检索（支持外部接口查询，失败回退本地测试 case）
- RAG 检索（必须使用 LlamaIndex）
- 代码检索（支持本地代码查询，或远程 Git 地址检索）

数据源策略：

- 默认 `external_first`（外部优先，失败回退本地）

## 2. 非目标

- 自动执行运维动作（重启、扩容、变更配置等）
- 多租户权限系统与复杂审批流
- 生产级高可用和大规模任务编排

## 3. 总体架构

采用模块化单体：

1. FastAPI API 层
2. LangGraph Orchestrator（状态机）
3. Planner（初始规划 + 重规划）
4. Skills（Log / RAG / Code / Final Synthesis）
5. Providers（External + Local fallback）
6. Memory（短期 + 长期）
7. Tracing / Observability / Evaluation

主调用路径：

`API -> LangGraph -> Skills Loop -> Judge/Replan -> Final LLM -> Response`

## 4. LangGraph 状态机设计（方案B）

### 4.1 节点

1. `parse_request`
2. `load_short_memory`
3. `plan_steps`
4. `execute_step`
5. `judge_next`
6. `retry_step`（条件分支）
7. `replan_if_needed`（条件分支）
8. `final_synthesis`
9. `compress_memory_if_needed`
10. `format_response`

### 4.2 核心状态 `AgentState`

- `request`: 用户请求及 source 偏好
- `parsed`: 结构化字段（traceId、orderId、service、keywords）
- `plan`: 当前步骤序列与策略
- `cursor`: 当前步骤索引
- `tool_results`: 原始工具返回
- `evidence`: 标准化证据集合（建议字段：`source/type/text/quality/timestamp`）
- `hypotheses`: 根因假设集合
- `decision`: continue/skip/retry/replan/finalize/stop
- `failure`: 最近失败信息（reason/action/attempt）
- `trace`: 节点执行轨迹
- `short_memory`: recent_turns + running_summary
- `final`: 最终输出

## 5. 失败分类、超时与恢复策略

### 5.1 枚举

`FailureReason`：

- `EXTERNAL_TIMEOUT`
- `EXTERNAL_5XX`
- `EXTERNAL_429`
- `EXTERNAL_AUTH_ERROR`
- `EXTERNAL_BAD_REQUEST`
- `NO_DATA_FOUND`
- `EVIDENCE_CONFLICT`
- `LLM_TIMEOUT`
- `LLM_NEED_MORE_INFO`
- `LLM_INVALID_OUTPUT`
- `UNKNOWN_ERROR`

`RecoveryAction`：

- `RETRY`
- `REPLAN`
- `STOP_FRIENDLY`

### 5.2 默认映射策略

- `EXTERNAL_TIMEOUT/EXTERNAL_5XX/EXTERNAL_429 -> RETRY`
- `NO_DATA_FOUND/EVIDENCE_CONFLICT/LLM_INVALID_OUTPUT -> REPLAN`
- `EXTERNAL_AUTH_ERROR/EXTERNAL_BAD_REQUEST/LLM_NEED_MORE_INFO -> STOP_FRIENDLY`
- `LLM_TIMEOUT -> RETRY（1次）后 STOP_FRIENDLY`

### 5.3 超时配置（默认）

- 外部日志 API：`connect=1s read=5s total=6s`
- 远程 Git 检索：`10s`
- RAG（LlamaIndex）：`8s`
- LLM 规划：`12s`
- LLM 最终生成：`20s`

## 6. 三类 Skill 与 Provider 适配器

### 6.1 LogSkill（外部优先）

- 主路径：`ExternalLogProvider`（HTTP/MCP）
- 回退：`LocalCaseLogProvider`（`demo_data/logs/*.json`）
- 输入：traceId/orderId/service/timeRange/keywords
- 输出证据：`LOG_ERROR/LOG_LATENCY/LOG_DOWNSTREAM`

### 6.2 RagSkill（LlamaIndex）

- Provider：`LlamaIndexRagProvider`
- 知识源：`src/rag_ingest/source_docs/system_docs` + `src/rag_ingest/source_docs/troubleshooting`
- 检索：关键词 + 向量检索（TopK 合并去重）
- 输出证据：`KB_CAUSE/KB_RUNBOOK/KB_ARCH_DEP`

### 6.3 CodeSkill（外部优先）

- 主路径：`GitRemoteCodeProvider`（git 地址）
- 回退：`LocalCodeProvider`（本地代码 + `rg`）
- 输入：service/module/error_keyword/stack_keyword
- 输出证据：`CODE_THROW_PATH/CODE_TIMEOUT_CONFIG/CODE_RETRY_POLICY`

## 7. Planner 与重规划规则

### 7.1 初始规划

- 有 traceId/orderId：优先 `logs`
- 常规默认：`logs -> rag -> code`

### 7.2 判定规则（`judge_next`）

- `finalize`：改为布尔规则组合，不再使用数值置信度阈值
- `continue`：本步有收获但证据仍不足
- `replan`：证据冲突、连续未命中、或失败映射为 REPLAN
- `stop`：失败映射为 STOP_FRIENDLY，或预算耗尽

`should_finalize(agent_state)` 规则：

```python
def should_finalize(agent_state: AgentState) -> bool:
    keywords = agent_state.parsed.keywords or agent_state.parsed.query_terms
    evidence_sources = {e.source for e in agent_state.evidence if e.quality == "high"}

    # 规则1: 交叉验证 - 至少两个不同来源的高质量证据
    cross_validated = len(evidence_sources) >= 2

    # 规则2: 无冲突 - 当前状态没有证据冲突
    no_conflict = (
        agent_state.failure is None
        or agent_state.failure.reason != FailureReason.EVIDENCE_CONFLICT
    )

    # 规则3: 覆盖性 - 证据文本可覆盖核心关键词
    covers_issue = any(
        keyword in e.text
        for e in agent_state.evidence
        for keyword in keywords
    )

    return cross_validated and no_conflict and covers_issue
```

### 7.3 重规划预算

- `replan_budget=2`
- `step_retry_max=2`（LLM timeout 特例 1 次）

## 8. Memory 设计（短期 + 长期）

### 8.1 短期记忆（会话内）

- 维护最近 `K` 轮完整对话（用户+模型）
- 超过阈值触发 `summary_update`，生成结构化摘要：
  - 背景
  - 已确认事实
  - 已尝试步骤
  - 未决问题
- 提示词注入方式：`recent_turns + running_summary`

默认建议：

- `K=8`
- token 水位阈值 `70% context`

### 8.2 长期记忆（跨会话）

MySQL 表（示意）：

- `id` BIGINT AUTO_INCREMENT PK
- `user_id` VARCHAR
- `session_id` VARCHAR
- `conversation_value` JSON（会话原始对话信息）
- `conversation_summary` TEXT
- `started_at` DATETIME
- `ended_at` DATETIME
- `created_at` DATETIME

写入时机：

- `sessions/end`
- `sessions/start` 时若有旧活跃会话，先归档旧会话

读取策略：

- 新会话加载最近 N 条长期摘要辅助规划，不默认回放全量原文

### 8.3 长期经验记忆（向量化）

在会话结束归档时，除写入 MySQL 外，增加经验提炼与向量化存储：

1. 经验提炼：
- 输入：本次会话 `evidence[]` 与 `final`
- 输出：1-3 条简洁经验描述
- 示例：`用户U123于2026-04-03遇到订单服务超时，根因是数据库连接池耗尽。`

2. 向量化与入库：
- 对每条经验描述生成 Embedding
- 写入向量库（可选 Qdrant/pgvector），并附 metadata：
  - `user_id`（必须）
  - `session_id`
  - `created_at`
  - `source="session_experience"`
  - `tags`（可选，如 `db_pool`, `timeout`）

3. 读取用途：
- 新会话规划前可检索该用户历史经验，作为 Planner 与 Final Synthesis 的补充上下文

## 9. API 契约（v1）

1. `POST /api/v1/analyze`
- 入参：`user_id, session_id, query, service_name, time_range, env`
- 可选：`log_source_hint, code_source_hint, git_repo_url`
- 出参：`root_cause, evidence[], suggestions[], confidence, trace_id, stop_reason?`
- `confidence` 类型：`"high" | "medium" | "low"`

2. `POST /api/v1/chat`
- 多轮对话入口，触发短期记忆加载和压缩

3. `POST /api/v1/sessions/start`
- 创建会话（必要时归档旧会话）

4. `POST /api/v1/sessions/end`
- 结束会话并写入长期记忆，同时执行经验提炼与向量入库

## 10. Prompt 契约

### 10.1 Planner Prompt（JSON 严格输出）

输出字段：

- `steps[]`
- `expected_evidence_per_step`
- `reasoning_brief`

### 10.2 Final Synthesis Prompt（JSON 严格输出）

输出字段：

- `root_cause`
- `evidence[]`（必须含来源）
- `suggestions[]`（仅人工执行建议）
- `confidence`（枚举：`high | medium | low`）
- `missing_info`

约束：

- 禁止编造证据来源
- 证据不足必须明确说明并列出补充信息
- 信心等级生成标准：
  - `high`：至少两个独立工具来源（如日志+知识库）提供强证据
  - `medium`：单一可靠来源强证据，或多来源弱证据
  - `low`：缺乏直接证据，主要依赖推理或假设

## 11. 可观测与评估

- Trace：节点输入摘要、输出摘要、耗时、异常、决策路径
- 指标：
  - `step_latency_ms`
  - `retry_count`
  - `replan_count`
  - `stop_friendly_count`
  - `external_fallback_rate`
- 评估：
  - `accuracy`
  - `completeness`
  - `actionability`

## 12. v1 交付里程碑

### 里程碑 A：主链路打通

- LangGraph 动态流程上线
- logs/rag/code 三 Skill 运行
- 外部优先 + fallback 生效

### 里程碑 B：记忆能力

- 短期记忆压缩
- 会话归档入 MySQL
- 会话经验提炼并写入向量库
- 新会话加载长期摘要

### 里程碑 C：稳定性与可观测

- 失败分类与恢复策略可视化
- trace + 指标完善
- 评估打分接入

## 13. 风险与缓解

- 外部接口不稳定：通过超时、重试、fallback 降级
- LLM 输出不稳定：强约束 JSON + 校验失败触发重规划
- 证据冲突：显式 `EVIDENCE_CONFLICT`，触发 replan 而非强行下结论

## 14. 验收标准（v1）

- 能处理“外部可用”和“外部不可用 fallback 本地”两类场景
- 可输出结构化 RCA（Root Cause/Evidence/Suggestion/Confidence）
- `confidence` 必须输出 `high/medium/low` 枚举值
- 具备可追踪执行轨迹
- 具备短期与长期记忆闭环
- 所有停止场景都有友好提示，不出现空响应
