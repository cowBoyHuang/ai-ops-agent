# RAG Parent Chunk Query Skill

名称: `rag_parent_chunk_query`

目标:
- 根据用户问题直接查询父 chunk TopK（内部自动完成子 chunk 排序），并加载完整父文档内容。
- 形成“业务结论 + 完整文档依据”的可复查证据链。

调用方法:
- `query_parent_docs_from_rag(question: str, intent_zh: str, sub_chunk_top_k: int | None = None, parent_top_k: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]`
  - 文件: `src/flow/modules/agent_executor_graph/graph/rag_retrieve/rag_retrieve.py`

参数说明:
- `question`(必填, string): 用户问题文本。
- `intent_zh`(必填, string): 中文意图标签（如“业务咨询”）。
- `sub_chunk_top_k`(可选, int): 内部子 chunk 候选数量，默认 6，最大 30。
- `parent_top_k`(可选, int): 父 chunk / 父文档返回数量，默认 4，最大 12。

返回要点:
- `sub_chunk_docs`: 内部排序得到的子 chunk 证据。
- `parent_chunk_docs`: 父 chunk TopK（每个父文档保留最高分子块）。
- `parent_docs`: 完整父文档内容（包含 `path`、`content`、`score`）。

使用规则:
1. 业务咨询场景仅对外暴露该能力，不再单独暴露子 chunk 查询入口。
2. 业务分析结论必须尽量引用 `parent_docs` 的完整上下文，避免只凭单个片段下结论。
3. 如果父文档无法读取，应明确标注“仅基于片段结论”。
