# System Workflow

用户输入问题：

订单创建失败

系统执行流程：

1. 检索知识库
2. Agent 生成排查计划
3. 调用 Skills
4. Skills 调用 Tools
5. LLM 分析问题
6. 输出 Root Cause

---

# 输出示例

Root Cause

数据库连接池耗尽导致订单创建失败

Evidence

日志中出现 DBConnectionTimeout

Suggestion

增加数据库连接池配置