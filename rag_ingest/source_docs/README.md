# 本地 RAG 文档导入目录

将排障文档放到本目录后，执行：

```bash
cd /Users/zhicheng.huang/code/qunar/ai-ops-agent
.venv/bin/python rag_ingest/import_local_docs.py
```

可选参数示例：

```bash
.venv/bin/python rag_ingest/import_local_docs.py \
  --source-dir /Users/zhicheng.huang/code/qunar/ai-ops-agent/rag_ingest/source_docs \
  --suffixes .md,.txt,.doc,.docx,.log,.rst
```

导入结果会写入本地 RAG 持久化目录（默认）：

- `/Users/zhicheng.huang/code/qunar/ai-ops-agent/.agent/rag/index`
