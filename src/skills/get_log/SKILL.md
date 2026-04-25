# Get Log Skill

名称: `get_log`

定位:
- 业务日志查询技能。
- 覆盖核心链路：生单、创单、支付前校验、生编、占座、支付等场景的日志检索与排障定位。

目标:
- 统一说明 `src/log` 下日志查询接口的入参与返参。
- 供 Planner/Tool 在生单、创单、支付前校验、生编、占座、支付等业务链路制定排障步骤时准确调用日志查询能力。

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

## appCode 与日志文件映射

1. `appCode: f_tts_trade_order`
- 系统说明: 生单系统，总调用入口，负责机票子单、营销子单、辅营子单的生单并调起各子单生单入口。
- 业务日志: `ttsorder.log`
- 异常日志: `ttsorder_error.log`

2. `appCode: f_tts_trade_core`
- 系统说明: 机票子单生单系统，负责创建机票子订单的实际实现。
- 业务日志: `tts.log`
- 异常日志: `tts_error.log`

## 调用建议

1. 先确定业务查询所需的 `appCode`、`logname`。
2. 确定 `appCode`、`logname` 后，根据实际订单号时间或 `traceId` 控制查询时间范围为前后一小时：
   - 以解析出的时间点 `T` 为中心，取 `begin_time = T - 1小时`，`end_time = T + 1小时`，并作为日志请求参数传入。
   - 订单号示例（固定格式）: `xep260425153507039`，可解析为 `2026-04-25 15` 点，按上述规则计算 `begin_time/end_time`。
   - `tradeId` 示例: `ops_slugger_260425.153507.10.95.136.249.868956.1362287667_1`，可解析为 `2026-04-25 15` 点，按上述规则计算 `begin_time/end_time`。
   - 同时参考“appCode 与日志文件映射”，将映射得到的 `app_code`、`logname` 作为日志查询参数传入。
3. 如果既没有可解析的订单号也没有可解析的 `tradeId`，查询时间需要结合上下文自行判断。
4. 其他查询字段（如 `content`、`type`）结合实际场景自行判断。
