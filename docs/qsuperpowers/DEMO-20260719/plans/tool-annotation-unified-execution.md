# Tool Annotation Unified Execution 实施计划

> **给智能代理工作者：** 必需：使用 qsuperpowers:subagent-driven-development（如果有子代理可用）或 qsuperpowers:executing-plans 来执行此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 将所有 Agent 可执行能力统一为 LangChain `@tool` 注册和调用，移除生产链路中的本地字符串 method 路由。

**架构：** 新增 `tool.registry` 作为唯一工具注册与调用入口；日志、Code Index、RAG 能力分别在专用模块中定义 `@tool` 函数。现有 planner/reactor/observer 图不重写，只替换 executor 和业务咨询节点的工具 schema 生成与调用路径。

**技术栈：** Python 3.11、LangChain `langchain_core.tools.tool`、LangGraph、pytest、Qdrant/RAG 现有模块、现有日志与 Code Index helper。

---

## 文件结构

- 创建: `src/tool/registry.py`，统一注册、查找、schema 生成、调用工具。
- 创建: `src/tool/log_tools.py`，定义 `queryLog`、`dependency_log_query`、`getFlightCreateOrderResult`、`getCreateOrderResult`。
- 创建: `src/tool/code_index_tools.py`，定义 `indexProject`、`searchMethod`、`locateCode`、`analyzeCodeFromLogs`、`analyzeCodeForBusinessConsult`。
- 创建: `src/tool/rag_tools.py`，定义 `rag_parent_chunk_query`、`knowledge_lookup`。
- 修改: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`，从 registry 生成 prompt schema 并执行工具。
- 修改: `src/flow/modules/agent_executor_graph/graph/executor/sub_executor/log_executor.py`，移除 `execute_log_query_method` 调用，改为 registry tool。
- 修改: `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`，将 legacy `log_method` 输入归一为注册 tool name，不再把 method 当本地路由键。
- 修改: `src/flow/modules/agent_executor_graph/graph/fixed_flow_execute/business_code_consult_skill.py`，改用 `analyzeCodeForBusinessConsult` tool。
- 修改: `src/log/log.py`、`src/log/__init__.py`，删除或停止导出 `execute_log_query_method`。
- 修改: `src/tool/code_index_client.py`、`src/tool/__init__.py`，删除或停止导出 `execute_code_index_method`。
- 修改: `src/llm/prompts/executor_react_system_prompt.txt`，删除 hard-coded legacy tool/method 指令。
- 修改: `src/llm/prompts/executor_react_user_prompt.txt`，要求模型直接选择注册 tool name。
- 测试: `src/tests/tool/test_tool_registry.py`。
- 测试: `src/tests/log/test_log_query_dispatch.py`。
- 测试: `src/tests/flow/test_code_index_client_trade_core.py`。
- 测试: `src/tests/flow/test_log_executor_dispatch.py`。
- 测试: `src/tests/flow/test_fixed_flow_execute.py`。
- 测试: `src/tests/flow/test_reactor.py`。

## 任务 1: Registry 和日志工具的失败测试

**文件：**

- 创建: `src/tests/tool/test_tool_registry.py`
- 修改: `src/tests/log/test_log_query_dispatch.py`

- [ ] **步骤 1: 编写 registry 失败测试**

在 `src/tests/tool/test_tool_registry.py` 添加：

```python
from __future__ import annotations

import datetime as dt

from log.log import EsResult
from tool.registry import build_tool_schemas_for_prompt, get_all_tools, get_tool, invoke_tool


EXPECTED_TOOLS = {
    "queryLog",
    "dependency_log_query",
    "getFlightCreateOrderResult",
    "getCreateOrderResult",
    "indexProject",
    "searchMethod",
    "locateCode",
    "analyzeCodeFromLogs",
    "analyzeCodeForBusinessConsult",
    "rag_parent_chunk_query",
    "knowledge_lookup",
}


def test_registry_loads_expected_tools_with_unique_names() -> None:
    names = [tool.name for tool in get_all_tools()]
    assert set(names) >= EXPECTED_TOOLS
    assert len(names) == len(set(names))


def test_prompt_schema_comes_from_tool_descriptions() -> None:
    schemas = build_tool_schemas_for_prompt()
    by_name = {row["tool_name"]: row for row in schemas}
    assert "机票子单生单结果" in by_name["getFlightCreateOrderResult"]["description"]
    assert "总单生单结果" in by_name["getCreateOrderResult"]["description"]
    assert "match_list=[]" in by_name["queryLog"]["description"]
    assert "业务文档" in by_name["analyzeCodeForBusinessConsult"]["description"]
    assert "实际代码" in by_name["analyzeCodeForBusinessConsult"]["description"]


def test_registry_invokes_query_log_tool(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_query_external_logs(**kwargs):
        captured.update(kwargs)
        return [EsResult(score=1.0, content="ok")]

    monkeypatch.setattr("log.log.query_external_logs", _fake_query_external_logs)
    begin = dt.datetime(2026, 1, 1, 11, 30, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    rows = invoke_tool(
        "queryLog",
        {
            "app_code": "f_tts_trade_order",
            "logname": "ttsorder",
            "begin_time": begin.isoformat(),
            "end_time": end.isoformat(),
            "match_phrase_list": ["ops_slugger_260101.120000.xxx"],
            "match_list": [],
        },
    )

    assert len(rows) == 1
    assert captured["app_code"] == "f_tts_trade_order"
    assert captured["logname"] == "ttsorder"
    assert get_tool("queryLog").name == "queryLog"


def test_invoke_tool_returns_structured_error_for_unknown_tool() -> None:
    out = invoke_tool("missingTool", {"x": 1})
    assert out == {
        "tool": "missingTool",
        "ok": False,
        "error": "unsupported tool: missingTool",
        "evidence": [],
    }
```

- [ ] **步骤 2: 重写日志工具测试为直接 tool 调用**

在 `src/tests/log/test_log_query_dispatch.py` 中移除 `execute_log_query_method` import，把测试改为 `invoke_tool()`：

```python
from tool.registry import invoke_tool

rows = invoke_tool(
    "getFlightCreateOrderResult",
    {
        "trace_id": "ops_slugger_260101.120000.xxx",
        "begin_time": begin.isoformat(),
        "end_time": end.isoformat(),
    },
)
```

保留现有断言：固定 app/log、生单固定短语、trace 从 `match_phrase_list` 继承。

- [ ] **步骤 3: 运行测试验证失败**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/tool/test_tool_registry.py tests/log/test_log_query_dispatch.py -q
```

预期: 失败，显示 `ModuleNotFoundError: No module named 'tool.registry'` 或缺少注册工具。

## 任务 2: 实现 `@tool` registry 和日志工具

**文件：**

- 创建: `src/tool/registry.py`
- 创建: `src/tool/log_tools.py`
- 修改: `src/tool/__init__.py`
- 修改: `src/log/log.py`
- 修改: `src/log/__init__.py`

- [ ] **步骤 1: 添加日志工具**

在 `src/tool/log_tools.py` 中添加 `@tool` 函数。实现必须调用现有 helper，不接收 `method` 参数。

```python
from __future__ import annotations

import datetime as dt

from langchain_core.tools import tool

from log.log import (
    LogApiConfig,
    dependency_log_query,
    get_create_order_result,
    get_flight_create_order_result,
    query_log,
)


@tool(
    "queryLog",
    description=(
        "通用底层日志查询技能。必须传 app_code、logname、begin_time、end_time、match_phrase_list。"
        "兜底 queryLog 必须满足 match_phrase_list 至少包含 traceId 或 orderNo，且 match_list=[]；"
        "match_phrase_list 只允许真实可落库检索的精确标识。"
    ),
)
def query_log_tool(
    app_code: str,
    logname: str,
    begin_time: dt.datetime | str,
    end_time: dt.datetime | str,
    match_phrase_list: list[str] | None = None,
    match_list: list[str] | None = None,
) -> list:
    return query_log(
        app_code=app_code,
        logname=logname,
        begin_time=begin_time,
        end_time=end_time,
        match_phrase_list=match_phrase_list or [],
        match_list=match_list or [],
    )
```

同文件继续添加：

- `dependency_log_query_tool`，tool name 为 `dependency_log_query`。
- `get_flight_create_order_result_tool`，tool name 为 `getFlightCreateOrderResult`，description 包含“机票子单生单结果”“固定 app/log”“不允许上层覆盖”。
- `get_create_order_result_tool`，tool name 为 `getCreateOrderResult`，description 包含“总单生单结果”“最终返回态”“聚合错误口径”“固定 app/log”。

不要在工具函数中接受 `method` 或 `log_method`。

同文件末尾必须导出：

```python
LOG_TOOLS = [
    query_log_tool,
    dependency_log_query_tool,
    get_flight_create_order_result_tool,
    get_create_order_result_tool,
]
```

- [ ] **步骤 2: 添加 registry**

在 `src/tool/registry.py` 中添加：

```python
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from tool.code_index_tools import CODE_INDEX_TOOLS
from tool.log_tools import LOG_TOOLS
from tool.rag_tools import RAG_TOOLS


_TOOLS: list[BaseTool] = [*LOG_TOOLS, *CODE_INDEX_TOOLS, *RAG_TOOLS]
_TOOL_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in _TOOLS}

if len(_TOOLS) != len(_TOOL_BY_NAME):
    names = [tool.name for tool in _TOOLS]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    raise RuntimeError(f"duplicate tool names: {duplicated}")


def get_all_tools() -> list[BaseTool]:
    return list(_TOOLS)


def get_tool(name: str) -> BaseTool:
    key = str(name or "").strip()
    if key not in _TOOL_BY_NAME:
        raise KeyError(f"unsupported tool: {key}")
    return _TOOL_BY_NAME[key]


def build_tool_schemas_for_prompt() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _TOOLS:
        schema = item.args_schema.model_json_schema() if item.args_schema is not None else {}
        rows.append(
            {
                "tool_name": item.name,
                "description": str(item.description or "").strip(),
                "params_schema": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        )
    return rows


def invoke_tool(name: str, args: dict[str, Any]) -> Any:
    key = str(name or "").strip()
    try:
        tool = get_tool(key)
    except KeyError:
        return {
            "tool": key,
            "ok": False,
            "error": f"unsupported tool: {key}",
            "evidence": [],
        }
    try:
        return tool.invoke(dict(args or {}))
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": key,
            "ok": False,
            "error": str(exc),
            "evidence": [],
        }
```

- [ ] **步骤 3: 添加临时空 code/RAG 工具列表以解开导入**

创建 `src/tool/code_index_tools.py` 和 `src/tool/rag_tools.py`，先导出空列表：

```python
from __future__ import annotations

CODE_INDEX_TOOLS = []
```

```python
from __future__ import annotations

RAG_TOOLS = []
```

- [ ] **步骤 4: 停止导出日志字符串路由**

从 `src/log/log.py` 删除 `execute_log_query_method()`。

从 `src/log/__init__.py` 删除 `execute_log_query_method` 的 import 和 `__all__` 项。

- [ ] **步骤 5: 更新 `src/tool/__init__.py`**

导出 registry 方法，不导出 `execute_code_index_method`：

```python
from tool.registry import build_tool_schemas_for_prompt, get_all_tools, get_tool, invoke_tool

__all__ = [
    "get_all_tools",
    "get_tool",
    "build_tool_schemas_for_prompt",
    "invoke_tool",
]
```

- [ ] **步骤 6: 运行日志和 registry 测试**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/tool/test_tool_registry.py tests/log/test_log_query_dispatch.py -q
```

预期: registry 测试仍因 Code Index/RAG 工具缺失失败；日志 direct tool 测试通过。

- [ ] **步骤 7: 提交日志工具基础**

```bash
git add src/tool/registry.py src/tool/log_tools.py src/tool/code_index_tools.py src/tool/rag_tools.py src/tool/__init__.py src/log/log.py src/log/__init__.py src/tests/tool/test_tool_registry.py src/tests/log/test_log_query_dispatch.py
git commit -m "feat: add annotated log tools registry"
```

## 任务 3: 实现 Code Index 和 RAG `@tool`

**文件：**

- 修改: `src/tool/code_index_tools.py`
- 修改: `src/tool/rag_tools.py`
- 修改: `src/tool/code_index_client.py`
- 修改: `src/tests/flow/test_code_index_client_trade_core.py`

- [ ] **步骤 1: 将 Code Index 测试改为直接 tool 调用**

在 `src/tests/flow/test_code_index_client_trade_core.py` 中移除 `execute_code_index_method` import。把 dispatcher 测试替换为：

```python
from tool.registry import invoke_tool


def test_search_method_tool_dispatches_search_method(self, monkeypatch) -> None:
    expected = _trade_core_method_row()
    monkeypatch.setattr(
        "tool.code_index_client.search_method",
        lambda keyword: {"ok": True, "methods": [expected], "error": ""},
    )

    out = invoke_tool("searchMethod", {"keyword": "getAsynRoundBackNote"})

    assert out["ok"] is True
    methods = list(out.get("methods") or [])
    assert str(dict(methods[0] or {}).get("methodName") or "") == "getAsynRoundBackNote"
```

同文件添加 `invoke_tool("analyzeCodeForBusinessConsult", {...})` 测试，保留原业务分析断言。删除 unknown method 路由测试。

- [ ] **步骤 2: 实现 Code Index tools**

在 `src/tool/code_index_tools.py` 中添加 `@tool` 包装：

```python
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from tool.code_index_client import (
    analyze_code_for_business_consult,
    analyze_code_from_logs,
    index_project,
    locate_code,
    search_method,
)


@tool("searchMethod", description="Code Index 方法搜索工具。根据关键词检索真实代码方法，返回类名、方法名、签名和文件行号。")
def search_method_tool(keyword: str) -> dict[str, Any]:
    return search_method(keyword)
```

同文件继续添加：

- `indexProject`，调用 `index_project(project_path)`。
- `locateCode`，调用 `locate_code(class_name, line)`。
- `analyzeCodeFromLogs`，调用 `analyze_code_from_logs(question, evidence_rows, extra_keywords)`。
- `analyzeCodeForBusinessConsult`，description 包含“业务咨询专用”“业务文档 + 实际代码分析双证据”“文档与代码冲突时以代码为准并标注文档待确认”。

导出：

```python
CODE_INDEX_TOOLS = [
    index_project_tool,
    search_method_tool,
    locate_code_tool,
    analyze_code_from_logs_tool,
    analyze_code_for_business_consult_tool,
]
```

- [ ] **步骤 3: 删除 Code Index 字符串路由**

从 `src/tool/code_index_client.py` 删除 `execute_code_index_method()`。

从 `src/tool/__init__.py` 确认没有导出 `execute_code_index_method`。

- [ ] **步骤 4: 实现 RAG tools**

在 `src/tool/rag_tools.py` 中添加：

```python
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from flow.modules.agent_executor_graph.graph.rag_retrieve.rag_retrieve import query_parent_docs_from_rag


@tool(
    "rag_parent_chunk_query",
    description="RAG 父文档检索工具。根据问题查询子 chunk TopK、父 chunk TopK，并加载完整父文档内容，形成可复查业务文档证据。",
)
def rag_parent_chunk_query_tool(
    query: str,
    intent_zh: str = "业务咨询",
    sub_chunk_top_k: int | None = None,
    parent_top_k: int | None = None,
) -> dict[str, Any]:
    sub_chunks, parent_chunks, parent_docs = query_parent_docs_from_rag(
        question=query,
        intent_zh=intent_zh,
        sub_chunk_top_k=sub_chunk_top_k,
        parent_top_k=parent_top_k,
    )
    return {
        "tool": "rag_parent_chunk_query",
        "ok": True,
        "error": "",
        "sub_chunks": sub_chunks,
        "parent_chunks": parent_chunks,
        "parent_docs": parent_docs,
    }
```

添加 `knowledge_lookup_tool(docs: list[dict[str, Any]] | None = None) -> dict[str, Any]`。它只从传入 docs 中提取前两条 content，不直接读全局 state；executor 负责传入当前 `knowledge_context.domain_docs`。

- [ ] **步骤 5: 运行 Code Index 和 registry 测试**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/tool/test_tool_registry.py tests/flow/test_code_index_client_trade_core.py -q
```

预期: 通过。

- [ ] **步骤 6: 提交 Code Index 和 RAG tools**

```bash
git add src/tool/code_index_tools.py src/tool/rag_tools.py src/tool/code_index_client.py src/tool/__init__.py src/tests/flow/test_code_index_client_trade_core.py src/tests/tool/test_tool_registry.py
git commit -m "feat: add annotated code and rag tools"
```

## 任务 4: 替换 executor、log executor、business consult 调用路径

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/executor/sub_executor/log_executor.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/fixed_flow_execute/business_code_consult_skill.py`
- 修改: `src/tests/flow/test_log_executor_dispatch.py`
- 修改: `src/tests/flow/test_fixed_flow_execute.py`

- [ ] **步骤 1: 重写 log executor 测试为 registry 调用断言**

在 `src/tests/flow/test_log_executor_dispatch.py` 中，把 monkeypatch 目标从 `execute_log_query_method` 改为 `tool.registry.invoke_tool` 或 `log_executor.invoke_tool`。

示例：

```python
def _fake_invoke_tool(name, args):
    captured["name"] = name
    captured.update(args)
    return [EsResult(score=1.0, content="line1")]

monkeypatch.setattr(
    "flow.modules.agent_executor_graph.graph.executor.sub_executor.log_executor.invoke_tool",
    _fake_invoke_tool,
)
```

断言改为：

```python
assert captured["name"] == "getCreateOrderResult"
assert "method" not in captured
assert "log_method" not in captured
```

对 legacy 输入用例保留 `params["log_method"]`，预期 executor 归一后调用注册 tool name。

- [ ] **步骤 2: 更新 `log_executor.py`**

导入：

```python
from tool.registry import invoke_tool
```

移除：

```python
from log.log import execute_log_query_method, resolve_log_method_scope
```

新增内部函数：

```python
_LOG_METHOD_TO_TOOL = {
    "querylog": "queryLog",
    "query_log": "queryLog",
    "log_query": "queryLog",
    "getcreateorderresult": "getCreateOrderResult",
    "get_create_order_result": "getCreateOrderResult",
    "getflightcreateorderresult": "getFlightCreateOrderResult",
    "get_flight_create_order_result": "getFlightCreateOrderResult",
    "dependency_log_query": "dependency_log_query",
    "query_dependency_log": "dependency_log_query",
}


def _resolve_registered_log_tool_name(tool_name: str, params: dict[str, Any]) -> str:
    raw = str(params.get("log_method") or tool_name or "queryLog").strip()
    return _LOG_METHOD_TO_TOOL.get(raw.lower(), raw)
```

在调用处构造 `tool_args`，删除 `method` 和 `log_method`：

```python
registered_tool = _resolve_registered_log_tool_name(tool_name, params)
tool_args = {
    "app_code": app_code,
    "logname": logname,
    "begin_time": begin_time,
    "end_time": end_time,
    "match_phrase_list": query_payload.get("match_phrase_list"),
    "match_list": query_payload.get("match_list"),
    "trace_id": dispatch_trace_id,
}
tool_args = {key: value for key, value in tool_args.items() if value is not None}
rows = invoke_tool(registered_tool, tool_args)
if isinstance(rows, dict) and rows.get("ok") is False:
    return {
        "tool": str(rows.get("tool") or registered_tool),
        "ok": False,
        "error": str(rows.get("error") or ""),
        "evidence": list(rows.get("evidence") or []),
    }
```

For `getCreateOrderResult` and `getFlightCreateOrderResult`, keep passing only `trace_id`, `begin_time`, and `end_time` unless additional parameters are harmless. These fixed-scope tools must not require upstream `app_code` or `logname`; they use their helper functions to apply fixed app/log internally.

After resolving `registered_tool`, update missing parameter handling:

```python
requires_upstream_scope = registered_tool in {"queryLog", "dependency_log_query"}
if requires_upstream_scope and (not app_code or not logname):
    # keep existing degraded fallback behavior
```

Remove all remaining uses of `fixed_scope`. There should be no `resolve_log_method_scope` import or local replacement of that function in `log_executor.py`.

- [ ] **步骤 3: 更新 executor schema 和 tool execution**

在 `executor.py` 中：

- Import `build_tool_schemas_for_prompt`, `get_all_tools`, `invoke_tool`.
- Replace `_ALLOWED_TOOLS` with `{tool.name for tool in get_all_tools()}`.
- Replace `_build_tool_schemas()` implementation with registry call or delete it and call `build_tool_schemas_for_prompt()`.
- Remove `_load_skill_catalog()` and stop passing `skill_catalog_json` into `executor_react_user_prompt.txt`. Prompt-facing executable descriptions must come from `build_tool_schemas_for_prompt()` only.
- Update aliases so `log_query` maps to `queryLog`, not to `log_query`.
- In `_execute_tool_call`, keep the log-specific sub-executor path only for registered log tool names: `queryLog`, `dependency_log_query`, `getCreateOrderResult`, `getFlightCreateOrderResult`.
- For `rag_parent_chunk_query`, invoke registry and adapt result into current evidence shape.
- For `knowledge_lookup`, call registry with `docs` from `state["knowledge_context"]["domain_docs"]`.
- For any `invoke_tool()` result where `ok is False`, return the structured error unchanged except for adding `tool` when absent.
- Wrap executor-boundary tool calls in a final `try/except` guard so unexpected exceptions still return `{"tool": tool_name, "ok": False, "error": str(exc), "evidence": []}`.

- [ ] **步骤 4: 更新 business consult skill**

Replace:

```python
from tool.code_index_client import execute_code_index_method
...
code_analysis = execute_code_index_method(
    method="analyze_code_for_business_consult",
    ...
)
```

With:

```python
from tool.registry import invoke_tool
...
code_analysis = invoke_tool(
    "analyzeCodeForBusinessConsult",
    {
        "question": str(question or "").strip(),
        "structured_context": dict(structured_context or {}),
        "evidence_rows": code_seed_rows,
    },
)
```

- [ ] **步骤 5: 运行 executor 相关测试**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/flow/test_log_executor_dispatch.py tests/flow/test_fixed_flow_execute.py -q
```

预期: 通过。

- [ ] **步骤 6: 提交 executor 集成**

```bash
git add src/flow/modules/agent_executor_graph/graph/executor/executor.py src/flow/modules/agent_executor_graph/graph/executor/sub_executor/log_executor.py src/flow/modules/agent_executor_graph/graph/fixed_flow_execute/business_code_consult_skill.py src/tests/flow/test_log_executor_dispatch.py src/tests/flow/test_fixed_flow_execute.py
git commit -m "feat: route executor through annotated tools"
```

## 任务 5: 清理 reactor legacy method 语义和旧路由引用

**文件：**

- 修改: `src/flow/modules/agent_executor_graph/graph/reactor/reactor.py`
- 修改: `src/flow/modules/agent_executor_graph/graph/executor/executor.py`
- 修改: `src/llm/prompts/executor_react_system_prompt.txt`
- 修改: `src/llm/prompts/executor_react_user_prompt.txt`
- 修改: `src/tests/flow/test_reactor.py`
- 修改: `src/tests/tool/test_tool_registry.py`
- 测试: `src/tests/flow/test_executor_tool_prompt.py`

- [ ] **步骤 1: 更新 reactor 测试期望**

在 `src/tests/flow/test_reactor.py` 中，将断言从 `final_params["log_method"] == "queryLog"` 改成工具名断言：

```python
assert str(final_action.get("tool_name") or "") == "queryLog"
assert "log_method" not in dict(final_action.get("params_summary") or {})
```

保留输入兼容测试：LLM 如果仍返回 `log_query + log_method=queryLog`，系统应归一到 `queryLog` tool name。

- [ ] **步骤 1.5: 添加 executor prompt 输入测试**

创建 `src/tests/flow/test_executor_tool_prompt.py`：

```python
from __future__ import annotations

import json

from flow.modules.agent_executor_graph.graph.executor import executor


def test_executor_prompt_uses_registry_tool_schemas_not_skill_catalog(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_render_prompt(template: str, **kwargs):
        captured.update({key: json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value for key, value in kwargs.items()})
        return "prompt"

    monkeypatch.setattr(executor, "render_prompt", _fake_render_prompt)
    monkeypatch.setattr(executor, "load_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(
        executor,
        "chat_with_llm",
        lambda **kwargs: '{"action":{"tool_name":"queryLog","params":{"match_phrase_list":["ops_slugger_260101.120000.xxx"],"match_list":[]}}}',
    )

    out = executor._decide_skill_with_llm(
        state={"question": "订单失败", "plan": {"hypothesis": "h", "investigation_goals": ["g"]}},
        hypothesis="h",
        objective="g",
        current_evidence={},
        evidence_rows=[],
        retry_count=0,
    )

    assert out["tool_name"] == "queryLog"
    rendered = json.dumps(captured, ensure_ascii=False)
    assert "tool_schemas_json" in captured
    assert "execute_log_query_method" not in rendered
    assert "execute_code_index_method" not in rendered
    assert "skill_catalog_json" not in captured
```

- [ ] **步骤 2: 更新 executor prompts**

在 `src/llm/prompts/executor_react_system_prompt.txt` 中删除或改写 hard-coded legacy 指令，例如：

- `tool_name=log_query|...`
- `params.log_method=queryLog`
- 要求通过 `log_method` 选择日志方法的描述。

替换为：

```text
action.tool_name 必须直接选择 tool_schemas_json 中存在的注册工具名。
日志工具必须直接选择 queryLog、getCreateOrderResult、getFlightCreateOrderResult 或 dependency_log_query。
不要输出 params.log_method 来选择能力。
```

在 `src/llm/prompts/executor_react_user_prompt.txt` 中删除要求模型输出 `log_method` 的指令。改为要求：

```text
action.tool_name 必须是 tool_schemas_json 中的 tool_name。
不要用 params.log_method 选择能力。
历史输入中的 log_method 仅用于兼容理解，新的 action 必须直接选择 queryLog、getCreateOrderResult、getFlightCreateOrderResult 或 dependency_log_query。
```

同时删除 prompt 模板中对 `skill_catalog_json` 的引用。模型不再读取原始 `SKILL.md` 文本来决定可执行方法。

- [ ] **步骤 3: 移除 runtime 中设置 `log_method` 的逻辑**

在 `executor.py` 和 `reactor.py` 中删除：

```python
tool_params.setdefault("log_method", ...)
```

用 `_normalize_tool_name()` 把 legacy alias 归一到注册工具名。保留兼容读取 `params["log_method"]`，但只用于确定 `tool_name`：

```python
def _normalize_legacy_tool_decision(tool_name: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidate = str(tool_name or "").strip()
    legacy_method = str(params.get("log_method") or "").strip()
    if candidate in {"log_query", "query_log"} and legacy_method:
        candidate = legacy_method
    normalized = _normalize_tool_name(candidate)
    cleaned = dict(params)
    cleaned.pop("log_method", None)
    return normalized, cleaned
```

- [ ] **步骤 4: 添加旧路由引用测试**

在 `src/tests/tool/test_tool_registry.py` 添加：

```python
from pathlib import Path


def test_production_runtime_has_no_old_string_router_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    production_files = [
        *root.glob("flow/**/*.py"),
        *root.glob("tool/**/*.py"),
        *root.glob("log/**/*.py"),
    ]
    offenders = []
    for path in production_files:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "execute_log_query_method" in text or "execute_code_index_method" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
```

- [ ] **步骤 5: 运行 reactor 和 no-router 测试**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/flow/test_reactor.py tests/tool/test_tool_registry.py -q
```

预期: 通过。

- [ ] **步骤 6: 提交 cleanup**

```bash
git add src/flow/modules/agent_executor_graph/graph/reactor/reactor.py src/flow/modules/agent_executor_graph/graph/executor/executor.py src/llm/prompts/executor_react_system_prompt.txt src/llm/prompts/executor_react_user_prompt.txt src/tests/flow/test_reactor.py src/tests/flow/test_executor_tool_prompt.py src/tests/tool/test_tool_registry.py
git commit -m "refactor: remove local method routing semantics"
```

## 任务 6: 全量 focused 回归和最终清理

**文件：**

- 修改: 仅修复前面测试发现的问题涉及的文件。

- [ ] **步骤 1: 搜索旧路由残留**

运行:

```bash
cd src && rg -n "execute_log_query_method|execute_code_index_method|params\\.get\\(\"log_method\"\\)|setdefault\\(\"log_method\"" flow tool log tests
```

预期:

- 生产代码中无 `execute_log_query_method` 或 `execute_code_index_method`。
- `log_method` 只允许出现在 legacy 兼容测试、提示词迁移说明或 `_normalize_legacy_tool_decision` 兼容函数内。

- [ ] **步骤 2: 运行 focused test suite**

运行:

```bash
cd src && ../.venv/bin/python -m pytest \
  tests/tool/test_tool_registry.py \
  tests/log/test_log_query_dispatch.py \
  tests/flow/test_code_index_client_trade_core.py \
  tests/flow/test_log_executor_dispatch.py \
  tests/flow/test_fixed_flow_execute.py \
  tests/flow/test_executor_tool_prompt.py \
  tests/flow/test_reactor.py \
  -q
```

预期: 全部通过。

- [ ] **步骤 3: 运行更宽的 flow/log/tool 回归**

运行:

```bash
cd src && ../.venv/bin/python -m pytest tests/log tests/flow tests/tool -q
```

预期: 全部通过；如果外部依赖测试因环境不可用失败，记录失败测试名和原因，不改动无关逻辑。

- [ ] **步骤 4: 检查 git diff**

运行:

```bash
git diff -- src/tool src/log src/flow src/tests src/llm/prompts/executor_react_system_prompt.txt src/llm/prompts/executor_react_user_prompt.txt
```

预期:

- Diff 只包含 `@tool` registry、调用替换、测试更新、prompt 去 method 语义。
- 没有无关格式化和 pycache 改动。

- [ ] **步骤 5: 最终提交**

```bash
git add src/tool src/log src/flow src/tests src/llm/prompts/executor_react_system_prompt.txt src/llm/prompts/executor_react_user_prompt.txt
git commit -m "feat: unify agent tools with annotations"
```

如果前面任务已按计划产生多个提交且没有剩余 diff，则跳过此提交。

## 完成标准

- `tool.registry.get_all_tools()` 返回所有规格列出的工具。
- executor prompt schemas 来自 `@tool` 对象。
- 主运行时不再调用 `execute_log_query_method` 或 `execute_code_index_method`。
- 业务咨询代码分析通过 `analyzeCodeForBusinessConsult` tool 调用。
- 日志工具不再通过 `method` 字符串选择实现。
- `queryLog` 的 trace/order 精确检索约束继续生效。
- Focused tests 通过，或任何环境性失败都有明确说明。
