# Project Structure

ai-ops-agent
│
├── docs
│
├── src
│
└── README.md

---

# docs

系统设计文档目录。

---

# src/rag_ingest/source_docs

本地 RAG 文档导入目录。

包含：

* 系统架构文档
* 故障排查案例

---

# src

核心代码目录。

模块包括：

* runtime（FastAPI 入口、路由、Web）
* core（agent、graph、llm、rag、providers、tools、utils）
* infra（session、memory、cache、tracing、evaluation、observability）
* rag_ingest（本地文档导入脚本与导入目录）
* tests（测试）
