---

```markdown
# AI Ops Agent

AI Ops Agent 是一个基于大语言模型（LLM）的智能运维分析系统，用于自动分析和定位系统问题，例如：

- 订单创建失败
- 接口调用超时
- 服务异常
- 错误日志分析
- 数据库连接问题

系统通过整合日志系统、代码仓库和运维知识库，自动完成问题分析并生成 Root Cause（根因分析）和排障建议。

---

## 项目目标

本项目旨在实现一个完整的 AI Agent 运维分析系统，具备以下能力：

- 自动理解用户问题
- 自动生成排查计划
- 自动查询日志系统
- 自动检索代码仓库
- 自动检索运维知识库
- 自动生成 Root Cause 分析
- 提供排障建议

项目重点不是简单调用 LLM，而是展示**完整的 AI Agent 系统架构设计**。

---

## 核心能力

### 自动问题分析

用户输入系统问题，例如：

> 订单创建失败，单号 ORD-20250330-1234

系统会自动执行：

1. **提取结构化标识**（如订单号、TraceID）
2. **优先从 Elasticsearch 精确查询对应日志**
3. 若无明确标识，则检索运维知识库
4. 生成问题排查计划
5. 检索相关代码（如有必要）
6. 分析错误原因
7. 输出 Root Cause

示例输出：

```
Root Cause
数据库连接池耗尽导致订单创建失败

Evidence
日志中出现 DBConnectionTimeout (TraceID: abc-123)

Suggestion
增加数据库连接池配置
```

---

## 系统核心模块

### 1. Agent Planner

负责根据用户问题生成排查任务计划。

示例任务计划：
- 若含 TraceID：`[fetch_logs_by_trace, root_cause_analysis]`
- 若无 TraceID：`[search_knowledge, analyze_logs, search_code, root_cause_analysis]`

### 2. Skills Layer

Skill 表示 Agent 可以执行的一种运维能力，例如：

- 知识库检索
- 日志分析（支持精确 TraceID 查询）
- 代码检索
- Root Cause 分析

结构如下：

```
Agent
↓
Skill
↓
Tool
```

### 3. Tool Calling

Tools 用于访问外部系统，例如：

- **Elasticsearch 日志查询**（支持 TraceID / 订单号精确匹配）
- Git 代码仓库检索
- 数据库查询
- 运维 API

Skill 调用 Tool 获取数据，再交由 LLM 进行分析。

### 4. RAG（Retrieval Augmented Generation）

系统通过 RAG 增强 LLM 的领域知识，但**优先利用结构化信息快速定位问题**。

**检索策略分两层**：

1. **精确日志召回（Primary Path）**  
   若用户输入包含 **订单号、TraceID、Error Code** 等结构化标识（如 “订单创建失败，单号 ORD-20250330-1234”），系统直接：
   - 从 Elasticsearch 中 **精确查询该 TraceID/订单号的全链路日志**
   - 提取关键错误片段（如 `DBConnectionTimeout`）
   - **跳过向量检索**，直接进入 LLM 分析阶段  
   → 实现 **秒级精准定位**，避免不必要的 RAG 开销。

2. **语义知识增强（Fallback / Context Enrichment）**  
   若无明确标识，或需补充上下文，则启动完整 RAG：
   - 使用 `BAAI/bge-small-zh` 对问题向量化
   - 在 Qdrant 中检索 Top-K 运维知识（Wiki、故障案例）
   - 结合 `BGE Reranker` 重排序，注入 LLM 上下文

> ✅ **优势**：  
> - 有 TraceID 时：**快、准、省 Token**  
> - 无 TraceID 时：**靠语义泛化兜底**

知识来源包括：
- 系统架构文档
- 运维 Wiki
- 历史排障案例
- 故障处理手册

RAG 工作流程（仅在无 TraceID 时触发）：
```
User Query
↓
Embedding (BGE Embeddings)
↓
Vector Search (Qdrant)
↓
Top-K Documents + BGE Reranker
↓
LLM Reasoning (OpenAI / DeepSeek / Qwen)
```

### 5. Memory

保存会话上下文，用于：

- 多轮问题分析
- 记录历史排查信息
- 支持连续排障对话

Memory 类型：**Session Memory（Redis）**

### 6. Cache

缓存 LLM 推理结果或 RAG 检索结果，减少成本和延迟。

- LLM Prompt → Response 缓存（相同问题不重复调用）
- RAG 检索结果缓存（相同 query 不重复向量搜索）

### 7. Model Router

根据任务类型选择不同模型：

- Agent Planning → GPT
- 日志分析 → DeepSeek
- 问答任务 → Qwen

优势：
- 降低成本
- 提高推理质量
- 提升系统灵活性

### 8. Reasoning Trace

记录完整的 Agent 推理过程：

- Agent Plan
- Skill Call
- Tool Call（含 ES 查询 DSL）
- Tool Result（含原始日志片段）
- LLM Reasoning
- Root Cause

示例：
```
Plan
fetch_logs_by_trace
root_cause_analysis

Skill Call
log_analysis_skill

Tool Call
ES query: {"term": {"trace_id": "abc-123"}}

Tool Result
{"log": "ERROR UserService: DBConnectionTimeout"}

Reasoning
数据库连接池耗尽

Root Cause
数据库连接池配置过小
```

### 9. Evaluation

系统支持 **人工和自动评估**：

- Accuracy：结果是否正确
- Completeness：是否覆盖关键问题
- Actionability：是否提供可执行建议

运维人员可在 Langfuse UI 对每次分析结果打标（✅/❌），用于持续优化 Agent 策略。

### 10. Observability

系统集成 **Langfuse** 实现端到端可观测性，支持：

- **完整推理 Trace 可视化**：
  - Agent Plan 步骤
  - Skill 调用链（如 `log_analysis_skill → elasticsearch_tool`）
  - Tool 输入/输出（如 ES 查询 DSL 与返回日志）
  - LLM Prompt 与生成结果
- **关键指标监控**：
  - LLM Latency / Token Usage（按模型分组）
  - Tool Success Rate（ES/Git 调用成功率）
  - Cache Hit Rate
- **人工评估闭环**：
  - 运维人员可在 Langfuse UI 对 Root Cause 打分（✅/❌）
  - 标注数据用于 A/B 测试不同 Agent 策略（如 “是否启用 reranker”）

> 🔧 **部署**：Langfuse 开源版私有化部署，数据不出内网。

---

## 系统架构

```
User
↓
API Layer (FastAPI)
↓
Memory (Redis Session)
↓
Cache (Redis)
↓
┌───────────────┐
│  Has TraceID? │
└───────┬───────┘
        ├─ Yes → Elasticsearch (Exact Match)
        └─ No  → RAG Retrieval (LlamaIndex + Qdrant + BGE Reranker)
                ↓
Agent Planner (LangGraph)
↓
Skills Layer
↓
Tool Execution (Elasticsearch + Git)
↓
Model Router (OpenAI / DeepSeek / Qwen)
↓
LLM Reasoning
↓
Evaluation (via Langfuse)
↓
Tracing & Observability (Langfuse)
```

---

## 技术栈

- **Agent Framework**: LangGraph + LangChain
- **RAG 框架**: LlamaIndex + BGE Embeddings (`bge-small-zh`) + Qdrant
- **重排序**: BGE Reranker (`bge-reranker-base`)
- **日志系统**: Elasticsearch（支持 TraceID 精确查询）
- **代码检索**: Git + LlamaIndex
- **LLM 推理**: OpenAI GPT / DeepSeek / Qwen（通过 Model Router 路由）
- **缓存与 Memory**: Redis
- **Web / API 层**: FastAPI
- **可观测性**: Langfuse（开源版私有部署）
- **流程控制**: LangGraph Planner + Executor + Skills

---

## 项目目录结构

```
ai-ops-agent/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── AGENT_DESIGN.md
│   ├── RAG_DESIGN.md
│   ├── MEMORY_DESIGN.md
│   ├── CACHE_DESIGN.md
│   ├── MODEL_ROUTER.md
│   ├── OBSERVABILITY.md          # ← 使用 Langfuse
│   ├── EVALUATION.md
│   ├── SYSTEM_WORKFLOW.md
│   └── AGENT_REASONING_TRACE.md
├── src/rag_ingest/source_docs/
│   ├── system_docs/
│   └── troubleshooting/
└── src/
    ├── api/
    ├── agent/
    ├── skills/
    ├── tools/
    ├── rag/
    ├── memory/
    ├── cache/
    ├── llm/
    ├── tracing/                  # ← 集成 Langfuse SDK
    ├── evaluation/
    ├── observability/
    └── utils/
```

---

## 示例执行流程

用户输入：
> 下单失败，TraceID: abc-123

系统执行：
1. **提取 TraceID `abc-123`**
2. **直接查询 Elasticsearch 获取完整日志**
3. Agent Planner 生成精简计划：`[fetch_logs_by_trace, root_cause_analysis]`
4. Skill 调用 ES Tool，返回日志片段
5. LLM 分析日志，生成 Root Cause
6. 结果与 Trace 自动上报至 **Langfuse**
7. 运维人员可在 Langfuse UI 审核并打标