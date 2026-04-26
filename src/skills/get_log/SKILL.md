# Get Log Skill

名称: `get_log`

定位:
- 业务日志查询技能，统一约束日志检索调用方式。

代码位置:
- 主文件: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/log.py`
- 导出入口: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/__init__.py`

## 核心协议（已更新）

日志查询条件统一通过两个列表传入，不再依赖 `type` 字段：

- `match_phrase_list: list[str]`  
  必须满足的精确短语条件（AND 关系，可为空）。
- `match_list: list[str]`  
  模糊匹配条件（OR 关系，可为空）。

至少有一个列表非空。

## 1) query_external_logs

签名:
```python
query_external_logs(
    *,
    app_code: str | None = None,
    logname: str = "",
    begin_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
    content: str | list[str] | dict[str, Any] = "",
    config: LogApiConfig | None = None,
    app_core: str | None = None,
    start_time: datetime | str | None = None,
    beginTime: datetime | str | None = None,
    endTime: datetime | str | None = None,
) -> list[EsResult]
```

推荐 `content` 结构:
```python
{
  "match_phrase_list": ["生单请求参数为", "ops_slugger_xxx"],
  "match_list": ["生单返回结果为", "traceId"]
}
```

返参:
- `list[EsResult]`，每个元素字段：
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
    content: str | list[str] | dict[str, Any],
    config: LogApiConfig | None = None,
) -> list[EsResult]
```

说明:
- 主流程封装：构建 ES 请求 -> 调接口 -> 结果适配。

## 3) build_es_pull_log_request

签名:
```python
build_es_pull_log_request(
    *,
    app_code: str,
    logname: str,
    begin_time: datetime | str,
    end_time: datetime | str,
    content: str | list[str] | dict[str, Any],
    max_lines: int = 1000,
) -> dict[str, Any]
```

`content` 在返回 payload 中会标准化为：
```python
{
  "match_phrase_list": [...],
  "match_list": [...]
}
```

## 4) pull_log_by_condition

签名:
```python
pull_log_by_condition(condition: dict[str, Any], config: LogApiConfig | None = None) -> dict[str, Any]
```

说明:
- 直接调用外部日志接口，HTTP/业务错误会抛异常。

## 5) 结果适配方法

签名:
```python
adapt_raw_response_to_es_results(raw: dict[str, Any]) -> list[EsResult]
adapt_raw_item_to_es_result(raw_item: Any) -> EsResult
```

## appCode 与日志文件映射

1. `appCode: f_tts_trade_order`
- 业务日志: `ttsorder`
- 异常日志: `ttsorder_error`

2. `appCode: f_tts_trade_core`
- 业务日志: `tts`
- 异常日志: `tts_error`

## 调用建议

1. 先确定 `app_code`、`logname`。
2. 时间范围优先按订单号/traceId 解析时间点 `T`，使用 `T-1h ~ T+1h`。
3. 条件建议：
   - `match_phrase_list` 放强约束（如 traceId、固定日志短语）
   - `match_list` 放扩召回关键词（如业务词、错误词）
