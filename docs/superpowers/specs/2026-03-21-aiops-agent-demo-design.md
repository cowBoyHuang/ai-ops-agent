# AIOps Agent Demo 设计文档

日期：2026-03-21

## 1. 背景

当前仓库已经具备 AIOps Agent 的概念架构文档，包括 Agent、Skills、Tools、RAG、Memory、Cache、Model Router、Tracing、Evaluation、Observability 等模块，但 `src/` 与 `src/rag_ingest/source_docs/` 目录尚未落地实际实现。

本设计文档的目标，是将这些概念模块收敛为一个第一期可运行 Demo 的完整设计，优先满足以下要求：

- 覆盖现有架构文档中提到的主要模块
- 提供真实的大模型调用能力
- 提供可演示的 HTTP API 和简单 Web 页面
- 保持模块边界稳定，便于后续替换真实日志、代码检索等外部系统

## 2. 第一期开发目标

第一期目标不是做生产级 AIOps 平台，而是完成一个“平台骨架型 Demo”：

- 采用 `Python + FastAPI`
- 提供 `HTTP API + 简单 Web 页面`
- 接入 `1 个真实 LLM`，同时为其他模型供应商保留配置占位
- 聚焦 `接口报错类故障`
- 重点保证 `可扩展性优先`

用户可以提交一个接口报错类问题，例如“order-service 下单失败”，系统通过规划、检索、工具调用和 LLM 汇总，返回结构化 Root Cause 分析、证据和建议，并在页面中展示分析链路与 Trace。

## 3. 非目标

第一期明确不做以下内容：

- 真实企业日志平台、代码仓库等外部系统的深度接入
- 自动修复和自动执行运维动作
- 复杂权限控制、多租户隔离
- 大规模异步任务调度和分布式执行
- 生产级高可用、水平扩展和资源治理

## 4. 方案选择

### 4.1 候选方案

#### 方案 A：单体直连型

所有模块放在一个服务中，模块边界较弱，优先追求尽快跑通。

优点：

- 开发速度快
- 调试路径短

缺点：

- 架构层次容易混乱
- 后续替换真实依赖成本高

#### 方案 B：模块化单体型

单服务部署，但在代码内部严格拆分模块边界、领域对象和适配层。

优点：

- 能完整落地现有架构文档中的模块
- 演示成本低于微服务
- 后续替换真实依赖顺滑

缺点：

- 需要先设计清楚接口契约
- 样板代码比方案 A 略多

#### 方案 C：微服务演示型

Agent、RAG、Tool、Trace 等拆成独立服务。

优点：

- 架构展示感强
- 服务边界清晰

缺点：

- 工程噪音大
- 第一期开销过高，容易挤压核心排障能力

### 4.2 选型结论

第一期采用 `方案 B：模块化单体型`。

理由：

- 能保留完整平台架构表达
- 能以较低复杂度打通端到端链路
- 便于后续按模块替换真实依赖或拆分服务

## 5. 总体架构

第一期 Demo 为单进程 FastAPI 服务，内部按模块拆分：

1. `API Layer`
2. `Web UI`
3. `Agent Orchestrator`
4. `Planner`
5. `Skills Layer`
6. `Tool Adapter Layer`
7. `RAG Service`
8. `Memory`
9. `Cache`
10. `Model Router`
11. `LLM Client Adapters`
12. `Tracing`
13. `Evaluation`
14. `Observability`

总体调用关系如下：

`Web/API -> Agent Orchestrator -> Planner -> Skills -> Tools/RAG/Memory/Cache/LLM -> Result -> Evaluation/Trace/Observability`

### 5.1 分层原则

- Agent 只负责编排，不直接访问外部系统
- Skill 代表业务能力，Tool 代表数据访问能力
- 所有外部依赖通过 Adapter 隔离
- Trace、Metrics、Evaluation 作为平台横切能力统一接入
- 模块优先通过结构化对象交互，而不是拼接自然语言字符串

## 6. 模块职责

### 6.1 API Layer

职责：

- 接收分析请求
- 管理 Session
- 返回结果、状态和 Trace

不负责：

- 直接做推理
- 直接访问日志或代码系统

### 6.2 Web UI

职责：

- 提供最小可演示页面
- 让用户提交问题与上下文
- 展示分析步骤时间线、证据、结论和评分

### 6.3 Agent Orchestrator

职责：

- 组织一次完整分析会话
- 调用 Planner 生成步骤
- 顺序或按依赖执行 Skills
- 聚合中间结果
- 调用 RCA Synthesizer 输出最终结论

### 6.4 Planner

职责：

- 将 `AnalysisRequest` 转为 `AnalysisPlan`
- 为接口报错类问题生成标准步骤序列

第一期默认步骤：

1. `knowledge_search`
2. `log_analysis`
3. `code_search`
4. `rca_synthesis`

### 6.5 Skills Layer

职责：

- 封装业务动作
- 将 Planner 的步骤映射为可执行逻辑

第一期建议 Skills：

- `KnowledgeSearchSkill`
- `LogAnalysisSkill`
- `CodeSearchSkill`
- `RcaSynthesisSkill`

### 6.6 Tool Adapter Layer

职责：

- 屏蔽外部系统差异
- 统一真实与 Mock 接口形式

第一期工具以 Mock 为主，但接口设计按真实系统抽象：

- `LogToolAdapter`
- `CodeToolAdapter`

### 6.7 RAG Service

职责：

- 对知识库文档做检索
- 向 Skills 或 RCA Synthesizer 提供上下文文档

第一期使用本地知识目录 + 向量检索或简化检索实现，保留升级空间。

### 6.8 Memory

职责：

- 保存 Session 级上下文
- 保存历史请求、步骤结果、最终结论

### 6.9 Cache

职责：

- 缓存 RAG 检索结果
- 缓存可复用的 LLM 输出

### 6.10 Model Router

职责：

- 根据任务类型路由到不同模型配置
- 第一期开启 1 个真实模型供应商，其他模型保留配置占位

### 6.11 LLM Client Adapters

职责：

- 对接真实模型服务
- 提供统一调用接口
- 屏蔽不同供应商请求和返回格式差异

### 6.12 Tracing

职责：

- 记录每个步骤的输入摘要、输出摘要、耗时和状态
- 保存关键中间结果与模型调用摘要

### 6.13 Evaluation

职责：

- 对分析结果做轻量质量评估
- 评估维度为 `accuracy / completeness / actionability`

### 6.14 Observability

职责：

- 记录系统运行指标和日志
- 支撑故障排查和演示

## 7. 核心领域对象

第一期必须定义稳定的数据模型。

### 7.1 AnalysisRequest

字段建议：

- `query`
- `service_name`
- `environment`
- `time_range`
- `symptom`
- `extra_context`

### 7.2 AnalysisSession

字段建议：

- `session_id`
- `status`
- `created_at`
- `request`
- `plan`
- `result`

### 7.3 AnalysisPlan

字段建议：

- `steps`
- `strategy`
- `planner_notes`

### 7.4 PlanStep

字段建议：

- `step_id`
- `name`
- `goal`
- `depends_on`
- `status`

### 7.5 Artifact

统一抽象不同来源的数据片段。

字段建议：

- `artifact_id`
- `type`
- `source`
- `content`
- `timestamp`
- `metadata`
- `raw_payload`

### 7.6 Evidence

字段建议：

- `evidence_id`
- `statement`
- `supported_by`
- `confidence`

### 7.7 AnalysisResult

字段建议：

- `root_cause`
- `confidence`
- `evidence`
- `suggestions`
- `next_actions`
- `summary`

### 7.8 SkillContext / SkillResult

统一 Skill 接口。

### 7.9 ToolQuery / ToolResult

统一 Tool Adapter 接口。

## 8. 请求链路设计

针对一次接口报错类问题，系统按如下流程执行：

1. 用户通过 Web 或 API 提交分析请求
2. API Layer 创建 `AnalysisSession`
3. Agent Orchestrator 调用 Planner 生成 `AnalysisPlan`
4. Skill Executor 按计划执行各个 Skill
5. Skill 通过 Tool Adapter、RAG、Memory、Cache 获取上下文与原始数据
6. 产出 `Artifact` 与 `Evidence`
7. `RcaSynthesisSkill` 调用真实 LLM 汇总中间结果
8. 形成 `AnalysisResult`
9. Evaluation 评估结果质量
10. Tracing 记录步骤过程
11. Observability 记录日志与指标
12. API 返回结构化结果，Web 展示时间线与详情

## 9. 外部依赖策略

### 9.1 LLM

第一期要求：

- 接入 1 个真实模型供应商
- 提供统一 `LLMClient` 接口
- 在配置中为其他供应商预留注册位

建议能力：

- 支持基础重试
- 支持超时控制
- 支持记录 token 使用量

### 9.2 业务工具

第一期采用 Mock Adapter，但接口形式必须贴近真实系统。

Mock 目标：

- 日志查询
- 代码片段查询

Mock 数据应覆盖至少 3 个接口报错类案例。

## 10. API 设计

第一期建议提供如下接口：

### 10.1 `POST /api/v1/analyze`

发起一次分析请求。

返回：

- `session_id`
- `status`
- `result`
- `trace_summary`

### 10.2 `GET /api/v1/sessions/{id}`

查询分析会话详情。

### 10.3 `GET /api/v1/sessions/{id}/trace`

查询完整 Trace，包括步骤状态、证据摘要和异常信息。

### 10.4 `GET /api/v1/chat`

检查服务可用性与关键依赖状态。

## 11. Web 页面设计

第一期 Web 页为最小演示页，建议由三部分组成：

1. 输入区
2. 分析过程时间线
3. RCA 结果与证据区

页面应支持：

- 输入问题、服务、环境和时间范围
- 提交分析请求
- 查看每个步骤的状态、耗时和摘要
- 查看最终 Root Cause、Evidence、Suggestion、Confidence
- 展开 Trace 查看中间 Artifact 摘要

## 12. 可靠性设计

### 12.1 Planner 失败

- 降级到预置排障模板
- 请求不直接失败

### 12.2 Tool 失败

- 标记步骤失败
- 尽量继续执行其他无依赖步骤
- 将失败信息写入 Trace

### 12.3 LLM 失败

- 支持重试
- 保留已收集 Artifact 和 Evidence
- 返回“证据不足/模型调用失败”的结构化结果

### 12.4 RAG 无命中

- 明确标记无知识命中
- 不伪造上下文

### 12.5 低置信度结果

- 不强行给出确定 Root Cause
- 改为返回 `next_actions`

## 13. 可观测性设计

### 13.1 Trace

每个步骤至少记录：

- `step_name`
- `start_time`
- `end_time`
- `latency_ms`
- `status`
- `input_summary`
- `output_summary`
- `error_summary`

### 13.2 Metrics

建议记录：

- 请求总数
- 会话成功率
- 各 Step 成功率
- LLM Latency
- Token Usage
- Cache Hit Rate
- Tool Error Rate

### 13.3 App Logs

应记录：

- 系统异常栈
- Tool Adapter 调用日志
- LLM 调用结果码
- 关键 Session 状态变化

## 14. 评估设计

Evaluation 不阻塞主请求，异步或后置执行均可。

第一期评估方式：

- 规则评估：检查输出结构完整性
- 轻量 LLM 评估：给出 `accuracy / completeness / actionability` 摘要

评估结果写入 Session 和 Trace，用于后续优化 Prompt 与检索策略。

## 15. 测试策略

### 15.1 单元测试

覆盖：

- Planner
- Model Router
- Skill 输入输出契约
- Tool Adapter
- Result Formatter

### 15.2 集成测试

覆盖：

- `POST /api/v1/analyze` 到结果返回的 happy path
- LLM 客户端适配层
- Trace、Evaluation、Observability 的串联

### 15.3 场景测试

至少准备 3 个接口报错类案例：

1. 数据库连接池耗尽
2. 下游接口 500 导致下单失败
3. 参数校验异常导致业务接口报错

每个案例应具备：

- Mock 日志
- Mock 代码片段
- 知识库条目
- 期望输出结果

## 16. 成功标准

第一期 Demo 完成后，应满足：

- 用户可以通过 API 或 Web 页面发起一次问题分析
- 系统可以返回结构化 RCA 结果
- 页面可展示计划步骤、证据和 Trace
- 真实 LLM 已接通
- 至少 3 个案例可演示
- 替换某个 Tool Adapter 或新增模型供应商时，不需要改 Agent 主流程

## 17. 实现建议顺序

建议实现顺序如下：

1. 目录与模块骨架
2. 核心领域模型
3. LLM Client 与 Model Router
4. Tool Adapter 与 Mock 数据
5. Planner / Skills / Orchestrator
6. API Layer
7. Web UI
8. Trace / Metrics / Evaluation
9. 测试与 Demo 样例

## 18. 风险与注意事项

- 如果过早追求真实外部系统接入，第一期交付会明显失控
- 如果不先定义结构化对象，后续很难扩展前端与平台能力
- 如果 Trace 与 Evaluation 不同步设计，Demo 会只剩“能回答”，缺少平台说服力
- 如果 Tool Adapter 抽象过浅，替换真实系统时会回流修改 Agent 主链路

## 19. 后续规划

第一期 Demo 完成后，下一阶段可以继续推进：

- 接入真实日志系统
- 增加真实代码检索接口
- 支持多轮追问
- 支持更复杂的问题类型，如超时和发版异常
