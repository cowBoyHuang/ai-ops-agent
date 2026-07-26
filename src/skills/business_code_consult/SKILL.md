# Business Code Consult Skill

名称: `business_code_consult`

定位:
- 业务咨询专用技能。
- 在回答业务问题前，必须补充“业务文档 + 实际代码分析（Code Index）”双证据。

外部依赖:
- 本地 Code Index Service: `http://127.0.0.1:18080`
  - `GET /searchMethod?keyword=...`
  - `GET /locateCode?class=...&line=...`

实现入口:
- Skill 执行入口: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/flow/modules/agent_executor_graph/graph/fixed_flow_execute/business_code_consult_skill.py`
- Code Index 统一分发: `/Users/zhicheng.huang/code/qunar/ai-ops-agent/src/tool/code_index_client.py`

调用方法:
- `execute_code_index_method(method="analyze_code_for_business_consult", ...)`
- 可选底层方法：
  - `execute_code_index_method(method="searchMethod", keyword=...)`
  - `execute_code_index_method(method="locateCode", class_name=..., line=...)`
  - `execute_code_index_method(method="indexProject", project_path=...)`

执行规则:
1. 先从业务文档（RAG）提取规则、系统边界与流程描述。
2. 再调用 Code Index 获取真实代码实现证据（方法定位/调用链）。
3. 最终输出时必须同时引用文档证据与代码证据；任一侧缺失时明确标注证据不足。
4. 文档与代码冲突时，优先说明“当前实现以代码为准，文档待确认/更新”。
