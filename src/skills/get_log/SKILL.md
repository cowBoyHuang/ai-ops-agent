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

### 强制规则（必须遵守）

当用户问题、上下文或已有参数中识别到以下字段时，必须把“实际识别到的值”写入 `match_phrase_list`：

- `traceId` / `trace_id`
- 完整 `ops_slugger` trace（例如 `ops_slugger_260506.110918.10.90.75.73.4022708.3131167276_1`）
- 订单号（如 `orderId` / `orderNo` / `订单号` / `子单号`）

约束：
- 这些精确标识禁止只放在 `match_list`，必须进入 `match_phrase_list`。
- `match_list` 仅用于模糊扩召回词，由执行器结合完整技能与上下文自行决定（例如“生单”“失败”“error”）。
- `match_phrase_list` 只允许真实、可落库检索的标识值；禁止写入占位符/不确定值（如 `ops_slugger_xxx`、`xxx`、`placeholder`、`tbd`、`todo`）。

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
  "match_phrase_list": ["ops_slugger_260506.110918.10.90.75.73.4022708.3131167276_1"],
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

## 固定格式示例（ops_slugger）

输入样例：
- `ops_slugger_260506.110918.10.90.75.73.4022708.3131167276_1`

时间解析规则：
- `260506` => `2026-05-06`
- `110918` => `11:09:18`
- 基准时间 `T = 2026-05-06T11:09:18+08:00`
- 推荐查询窗：`T-1h ~ T+1h`
  - `begin_time = 2026-05-06T10:09:18+08:00`
  - `end_time = 2026-05-06T12:09:18+08:00`

推荐查询参数（先查 trade_order）：
```python
query_external_logs(
    app_code="f_tts_trade_order",
    logname="ttsorder.log",
    begin_time="2026-05-06T10:09:18+08:00",
    end_time="2026-05-06T12:09:18+08:00",
    content={
        "match_phrase_list": ["ops_slugger_260506.110918.10.90.75.73.4022708.3131167276_1"],
        "match_list": ["订单创建失败", "生单失败", "Exception", "ERROR"],
    },
)
```

依赖侧查询（trade_core）示例：
```python
query_external_logs(
    app_code="f_tts_trade_core",
    logname="tts.log",
    begin_time="2026-05-06T10:09:18+08:00",
    end_time="2026-05-06T12:09:18+08:00",
    content={
        "match_phrase_list": ["ops_slugger_260506.110918.10.90.75.73.4022708.3131167276_1"],
        "match_list": ["特殊产品拦截", "子单失败", "errorCode", "timeout"],
    },
)
```

## 日志不足时的代码分析兜底（V2）

当日志无法直接给出结论时（例如命中为空、仅有模糊失败描述、缺少明确错误码/失败原因）：

1. 必须触发本地 Code Index Client 做代码上下文补充：
   - `GET /locateCode?class=...&line=...`
   - `GET /searchMethod?keyword=...`
2. 优先使用日志中的类名+行号定位方法；定位失败再用关键词搜索方法。
3. 代码兜底结果必须并入同一轮 `evidence/effective_info`，由后续 LLM 综合分析。
4. 若 Code Index 不可达或未命中，不得伪造代码结论，需明确写出失败原因。
