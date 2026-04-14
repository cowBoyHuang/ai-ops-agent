# Agent Design

Agent 负责自动执行问题分析流程。

---

# Agent 工作流程

用户输入问题：

订单创建失败

Agent 自动执行：

1. 检索运维知识
2. 查询日志
3. 查询代码
4. LLM 分析原因
5. 输出 Root Cause

---

# Agent Planner

Agent 首先生成任务计划，例如：

1. search_knowledge
2. analyze_logs
3. search_code
4. root_cause_analysis

---

# Skills 调用

Agent 不直接调用 Tool，而是调用 Skill。

例如：

log_analysis_skill  
code_search_skill  
knowledge_search_skill  

---

# LLM Reasoning

LLM 根据 Skill 返回结果进行综合分析。

---

# 输出

最终输出：

- Root Cause
- Evidence
- Suggestion