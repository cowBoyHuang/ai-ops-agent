# Get Flight Create Order Result Skill

名称: `getFlightCreateOrderResult`

定位:
- 机票子单生单结果专用日志技能（面向“机票子单执行与校验链路”的细粒度视角）。
- 对应机票生单主流程中的校验/路由/创建阶段，适合定位失败发生点与业务错误码来源。
- 对外屏蔽 `app_code` 与 `logname`，在实现中固定为:
  - `app_code = f_tts_trade_core`
  - `logname = tts`

调用方法:
- `execute_log_query_method(method=\"getFlightCreateOrderResult\", ...)`

入参:
- `trace_id`（建议必传）
- `begin_time`
- `end_time`

实现约束:
- 技能内部固定拼接短语: `单程生单结果`
- 不允许由上层覆盖固定的 `app_code/logname`

业务场景:
- 场景1：用户问“机票子单为什么失败/失败发生在什么校验环节”，需要定位子单内部失败语义时使用。
- 场景2：总单结果已确定失败，但需要进一步下钻到机票生单链路（规则校验、路由、创建）时使用。
- 场景3：需要确认失败是否来自机票业务规则（如乘机人规则、产品专项规则）时使用。
- 典型触发：总单只给出聚合失败口径，无法解释机票侧失败来源。

与其他技能关系:
- 本技能负责“机票子单链路细节与失败归因”。
- `getCreateOrderResult` 负责“最终返回态与聚合口径”。
- `queryLog` 负责“当子单链路仍不足时的扩展兜底检索”。

代码入口:
- `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/log.py`
