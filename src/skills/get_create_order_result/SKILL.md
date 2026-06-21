# Get Create Order Result Skill

名称: `getCreateOrderResult`

定位:
- 总单生单结果专用日志技能（面向“最终返回给调用方”的结果视角）。
- 最终返回给用户的生单结果（包括机票生单、营销生单、辅营生单的聚合结果）
- 对应业务链路中的“结果修复与汇总层/响应封装返回”阶段，适合确认最终成功/失败结论与聚合后的错误信息。
- 对外屏蔽 `app_code` 与 `logname`，在实现中固定为:
  - `app_code = f_tts_trade_order`
  - `logname = ttsorder`

调用方法:
- `execute_log_query_method(method=\"getCreateOrderResult\", ...)`

入参:
- `trace_id`（建议必传）
- `begin_time`
- `end_time`

实现约束:
- 技能内部固定拼接短语: `生单返回结果`
- 不允许由上层覆盖固定的 `app_code/logname`

业务场景:
- 场景1：用户问“这笔单为什么失败/最终返回了什么”，需要先判断主链路最终返回态时使用。
- 场景2：已经知道可能是机票生单问题，但需要确认“是否在总单汇总返回阶段被判失败”时使用。
- 场景3：需要对外口径（最终失败提示、聚合后错误语义）时使用。
- 非首选场景：仅追问机票子单内部细节或特定业务码来源时，不应只依赖本技能。

与其他技能关系:
- 本技能负责“最终态与返回口径”。
- `getFlightCreateOrderResult` 负责“机票子单执行链路细节”。
- `queryLog` 负责“跨服务/跨关键词兜底扩展检索”。

代码入口:
- `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/log.py`
