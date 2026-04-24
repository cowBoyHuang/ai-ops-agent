# Get Log Skill

名称: `get_log`

目标:
- 统一说明 `src/log` 下日志查询接口的入参与返参。
- 供 Planner/Tool 在制定排障步骤时准确调用日志查询能力。

代码位置:
- 主文件: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/log.py`
- 导出入口: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/__init__.py`

## 1) query_external_logs

签名:
```python
query_external_logs(
    *,
    app_code: str | None = None,
    logname: str = "",
    begin_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
    content: str = "",
    type: str = "",
    config: LogApiConfig | None = None,
    app_core: str | None = None,
    start_time: datetime | str | None = None,
    beginTime: datetime | str | None = None,
    endTime: datetime | str | None = None,
) -> list[EsResult]
```

关键入参:
- `app_code` / `app_core`: 应用标识，至少一个必填。
- `logname`: 日志名（必填）。
- `begin_time`, `end_time`: 时间范围（必填，支持 datetime 或字符串）。
- `content`: 查询关键词（必填）。
- `type`: 查询类型，支持 `match` 或 `match_phrase`。
- `config`: 可选日志接口配置，默认 `LogApiConfig.from_env()`。

返参:
- `list[EsResult]`，每个元素字段:
  - `score: float`
  - `content: str`

## 2) search_logs

签名:
```python
search_logs(
    *,
    app_code: str,
    logname: str,
    begin_time: datetime | str,
    end_time: datetime | str,
    content: str,
    type: str,
    config: LogApiConfig | None = None,
) -> list[EsResult]
```

说明:
- 对外日志查询主流程封装（构建 ES 请求 -> 调接口 -> 适配结果）。

返参:
- `list[EsResult]`（同上）。

## 3) build_es_pull_log_request

签名:
```python
build_es_pull_log_request(
    *,
    app_code: str,
    logname: str,
    begin_time: datetime | str,
    end_time: datetime | str,
    content: str,
    type: str,
    max_lines: int = 200,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]
```

返参:
- ES 查询请求体字典，包含:
  - `index`, `body`
  - `appCode`, `beginTime`, `endTime`, `logname`, `content`, `type`

## 4) pull_log_by_condition

签名:
```python
pull_log_by_condition(condition: dict[str, Any], config: LogApiConfig | None = None) -> dict[str, Any]
```

入参:
- `condition`: 查询条件（可为完整请求或含 `body/index` 的结构）。
- `config`: 可选配置。

返参:
- 原始响应 `dict[str, Any]`。
- 业务失败会抛异常（如 HTTP 错误、URL 错误、business error）。

## 5) 结果适配方法

签名:
```python
adapt_raw_response_to_es_results(raw: dict[str, Any]) -> list[EsResult]
adapt_raw_item_to_es_result(raw_item: Any) -> EsResult
```

说明:
- 把原始日志接口返回适配成统一 `EsResult` 列表。

## QueryType

枚举:
- `QueryType.MATCH = "match"`
- `QueryType.MATCH_PHRASE = "match_phrase"`

## 调用建议

1. 常规查询优先使用 `query_external_logs`。
2. 需要自定义请求结构时，先 `build_es_pull_log_request`，再 `pull_log_by_condition`。
3. 统一把返回结果适配为 `list[EsResult]` 再进入后续流程。

