# System Architecture

系统整体架构如下：

User  
↓  
API Layer  
↓  
Memory  
↓  
Cache  
↓  
RAG Retrieval  
↓  
Agent Planner  
↓  
Skills Layer  
↓  
Tool Execution  
↓  
Model Router  
↓  
LLM Reasoning  
↓  
Evaluation  
↓  
Tracing  
↓  
Observability  

---

# 核心模块

## API Layer

提供统一接口接收用户问题。

---

## Memory

保存对话上下文。

---

## Cache

缓存 LLM 和 RAG 结果。

---

## RAG Retrieval

从知识库检索相关文档。

---

## Agent Planner

生成问题分析计划。

---

## Skills Layer

Skill 表示 Agent 可以执行的一种能力，例如：

- 日志分析
- 代码检索
- 知识检索
- Root Cause 分析

Skill 可以组合多个 Tool。

---

## Tool Execution

调用外部系统：

- Elasticsearch
- Git Repository
- Database

---

## Model Router

根据任务选择不同模型。

---

## LLM Reasoning

使用 LLM 对问题进行分析推理。

---

## Evaluation

评估回答质量。

---

## Tracing

记录 Agent 推理轨迹。

---

## Observability

记录系统运行指标。