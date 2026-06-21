# Query Log Skill

名称: `queryLog`

定位:
- 通用底层日志查询技能。
- 复用 `src/skills/get_log` 中的查询协议与规则。

依赖关系:
- 规则与参数规范: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/skills/get_log/SKILL.md`
- 实现入口: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/log/log.py`

调用方法:
- `execute_log_query_method(method=\"queryLog\", ...)`

必须由上层传入:
- `app_code`
- `logname`
- `begin_time`
- `end_time`
- `match_phrase_list`

说明:
- `queryLog` 不写死业务系统参数，用于兜底与扩展场景。
- 兜底 `queryLog` 必须满足：
  - `match_phrase_list` 至少包含一个 `traceId` 或 `orderNo`（可从上下文继承）。
  - `match_list` 必须为空，不允许模糊扩召回词。
