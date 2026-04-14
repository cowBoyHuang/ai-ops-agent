# 📚 RAG 系统设计文档（运维领域增强版）

> **目标**：构建一个面向运维场景的高精度、低延迟 RAG 系统，支持从 Wiki、架构文档、历史故障日志中自动检索相关信息，辅助 LLM 生成准确诊断建议。

---

## 🔗 1. 知识来源（真实数据接入）

| 数据类型 | 存储系统 | 接入方式 | 更新频率 |
|--------|--------|--------|--------|
| **运维 Wiki** | Confluence / Markdown 文件 | 定期导出 + LlamaIndex `SimpleDirectoryReader` | 每日 |
| **系统架构文档** | Git 仓库（PDF/Markdown） | `LlamaParse`（PDF） + `MarkdownReader` | 每周 |
| **历史故障案例** | **Elasticsearch (ES)** | 自定义 `ElasticsearchReader`（基于 LlamaIndex） | 实时/每小时 |
| **应用日志（错误/告警）** | **Elasticsearch (ES)** | 按时间窗口拉取（如最近 7 天 ERROR 日志） | 实时 |

> ✅ **关键优势**：  
> 直接对接生产 ES，无需额外 ETL，确保知识库**实时反映线上状态**。

---

## 🧠 2. 整体架构（LlamaIndex 驱动）

```mermaid
graph LR
    A[User Query] --> B(LlamaIndex Query Engine)
    B --> C{检索策略}
    C --> D[Qdrant 向量检索]
    C --> E[ES 关键词检索（可选）]
    D --> F[BGE Reranker 重排序]
    E --> F
    F --> G[Top-K Context]
    G --> H[LLM 生成答案]
    H --> I[返回诊断建议]
    
    subgraph Data Ingestion
        J[Wikis/Docs] --> K[LlamaIndex Ingestion Pipeline]
        L[Elasticsearch Logs] --> K
        K --> M[Qdrant Vector Store]
    end