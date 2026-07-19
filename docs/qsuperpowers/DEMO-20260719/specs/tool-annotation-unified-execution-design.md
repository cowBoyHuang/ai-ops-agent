# Tool Annotation Unified Execution Design

## Background

The current agent runtime exposes several executable capabilities through local string dispatchers:

- `log.log.execute_log_query_method(method=...)` routes log methods such as `queryLog`, `getCreateOrderResult`, and `getFlightCreateOrderResult`.
- `tool.code_index_client.execute_code_index_method(method=...)` routes Code Index methods such as `searchMethod`, `locateCode`, and business code analysis.
- `executor._build_tool_schemas()` maintains a separate hand-written prompt schema list.
- Skill documents under `src/skills/**/SKILL.md` describe business usage rules, but those descriptions are not the executable tool source of truth.

This creates duplicate routing logic and allows tool descriptions, skill documents, and executable functions to drift.

## Goal

Unify all executable agent capabilities behind LangChain `@tool` annotations. The runtime must invoke registered tool objects directly and must not use local string-based method routers.

## Non-Goals

- Do not replace the existing planner, reactor, observer, or LangGraph flow.
- Do not introduce a generic LangChain agent loop.
- Do not redesign RAG retrieval quality, chunking, or reranking.
- Do not change external log service or Code Index service protocols.

## Design

### Tool Registry

Add `src/tool/registry.py` as the single executable tool registry.

Responsibilities:

- Import all annotated tools.
- Validate tool names are unique.
- Expose `get_all_tools() -> list[BaseTool]`.
- Expose `get_tool(name: str) -> BaseTool`.
- Expose `build_tool_schemas_for_prompt() -> list[dict[str, Any]]`.
- Expose `invoke_tool(name: str, args: dict[str, Any]) -> Any`.

The registry is the only place that maps a model-provided `tool_name` to an executable tool object. It maps tool names, not business method strings.

### Annotated Tool Modules

Create dedicated modules for executable tools:

- `src/tool/log_tools.py`
- `src/tool/code_index_tools.py`
- `src/tool/rag_tools.py`

Each public executable capability is a separate `@tool` function. The tool name is the method identity. There is no `method` parameter for selecting behavior.

Required log tools:

- `queryLog`
- `dependency_log_query`
- `getFlightCreateOrderResult`
- `getCreateOrderResult`

Required Code Index tools:

- `indexProject`
- `searchMethod`
- `locateCode`
- `analyzeCodeFromLogs`
- `analyzeCodeForBusinessConsult`

Required knowledge tools:

- `rag_parent_chunk_query`
- `knowledge_lookup`

Existing helper functions such as `query_log`, `get_create_order_result`, `search_method`, and `locate_code` may remain as implementation helpers. They are not routers and do not select behavior based on a method string.

### Skill Descriptions Become Tool Descriptions

Move the actionable content from matching `SKILL.md` files into `@tool` descriptions:

- Tool purpose and business scenario.
- Required parameters.
- Fixed app/log scope, when applicable.
- Constraints such as `queryLog` requiring precise trace/order identifiers and `match_list=[]` for fallback.
- Relationship between specialized tools and generic tools.
- Evidence requirements for business code consultation.

Skill files may remain as human-readable documentation, but the prompt-facing tool description comes from the annotated tool object.

### Remove Local String Routers

Remove these routing entry points from the main runtime:

- `execute_log_query_method`
- `execute_code_index_method`
- Any executor branch that chooses behavior by a local `method` string.

Call sites must invoke `tool.registry.invoke_tool(tool_name, params)` or a retrieved `BaseTool.invoke(params)`.

The model may still output `tool_name`, but it must not output `method` to select an implementation. If legacy prompts or code produce `method`, executor may copy it into `tool_name` only as a temporary input normalization step before registry lookup. It must not use `method` for local dispatch.

### Executor Integration

Update `flow/modules/agent_executor_graph/graph/executor/executor.py`:

- Replace `_build_tool_schemas()` with `tool.registry.build_tool_schemas_for_prompt()`.
- Keep the allowed-tool safety boundary by deriving allowed names from the registry.
- Normalize aliases only to registered tool names.
- Execute tools through the registry.
- Preserve existing evidence graph output shape.

Update `reactor.py` and `log_executor.py` only where necessary to pass a registered `tool_name` and params.

### Business Code Consult Integration

Update `business_code_consult_skill.py`:

- Replace `execute_code_index_method(method="analyze_code_for_business_consult", ...)` with registry invocation of `analyzeCodeForBusinessConsult`.
- Preserve the current evidence merge behavior.
- Preserve the rule that final analysis uses both business document evidence and Code Index evidence.

### Error Handling

Registry lookup failure returns a structured tool error:

```python
{
    "tool": "<name>",
    "ok": False,
    "error": "unsupported tool: <name>",
    "evidence": [],
}
```

Tool execution exceptions are caught at the executor boundary and returned in the same structured shape.

Parameter validation remains split:

- Tool argument schema validates required shape.
- Reactor/executor guardrails validate business constraints such as precise identifiers, time windows, and fallback `queryLog` restrictions.

### Testing

Add or update tests for:

- Registry loads all expected tools.
- Tool names are unique.
- Prompt schema is generated from `@tool` objects, not a hand-written schema list.
- `log_executor` and executor do not import or call `execute_log_query_method`.
- `business_code_consult_skill` does not import or call `execute_code_index_method`.
- Each former string-routed method is directly invokable by registered tool name.
- Tool descriptions include the corresponding skill business rules.
- Existing queryLog guardrails still reject missing precise identifiers and fuzzy fallback terms.

## Migration Steps

1. Add annotated tool modules and registry.
2. Move skill method descriptions into `@tool` descriptions.
3. Replace executor schema generation and invocation with registry calls.
4. Replace business code consult Code Index invocation with registry calls.
5. Remove local string routers from production call paths.
6. Update tests to validate direct tool execution and absence of router calls.
7. Run focused unit tests for registry, log dispatch, Code Index, fixed-flow business consult, and reactor/executor guardrails.

## Acceptance Criteria

- All executable agent capabilities are represented as LangChain `@tool` functions.
- No production runtime path calls `execute_log_query_method` or `execute_code_index_method`.
- No production runtime path chooses tool behavior by matching a local `method` string.
- Executor prompt tool schemas are generated from registered tools.
- Tool descriptions contain the business constraints currently documented in the corresponding skill files.
- Existing AIOps graph behavior remains intact: planner, reactor, observer, replan, and final answer flow are unchanged.
- Focused tests pass for tool registry, log tools, Code Index tools, business consult, and executor integration.
