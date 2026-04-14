# Agent Reasoning Trace

系统会记录 Agent 推理过程。

---

# Trace 内容

- Agent Plan
- Skill Call
- Tool Call
- Tool Result
- LLM Reasoning
- Root Cause

---

# 示例

Plan

search_knowledge  
analyze_logs  

Skill Call

log_analysis_skill

Tool Result

日志中出现 DBConnectionTimeout

Reasoning

数据库连接池耗尽

Root Cause

数据库连接池配置过小