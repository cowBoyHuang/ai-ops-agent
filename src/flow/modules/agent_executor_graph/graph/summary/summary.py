"""Investigation summary node."""

from __future__ import annotations

import json
from typing import Any

from flow.modules.agent_executor_graph.agent_state import AgentState
from llm.llm import analyze_with_llm


def _build_evidence_text(investigation: dict[str, Any]) -> str:
    rows: list[str] = []
    for idx, item in enumerate(list(investigation.get("evidence") or []), start=1):
        row = dict(item or {})
        facts = dict(row.get("facts") or {})
        evidence_items = list(row.get("evidence") or [])
        rows.append(
            "\n".join(
                [
                    f"[goal#{idx}] id={row.get('goal_id')} executor={row.get('executor')} status={row.get('status')}",
                    f"summary={row.get('summary')}",
                    f"facts={json.dumps(facts, ensure_ascii=False)}",
                    f"evidence={json.dumps(evidence_items[:5], ensure_ascii=False, default=str)}",
                ]
            )
        )
    return "\n\n".join(rows).strip()


def run(payload: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = dict(payload)
    investigation = dict(state.get("investigation") or {})
    plan = dict(investigation.get("plan") or {})
    evidence_text = _build_evidence_text(investigation)
    if evidence_text:
        analysis = analyze_with_llm(
            question=str(state.get("question") or ""),
            evidence="\n".join(
                [
                    f"hypothesis={str(plan.get('hypothesis') or '')}",
                    f"finish_criteria={json.dumps(list(plan.get('finish_criteria') or []), ensure_ascii=False)}",
                    evidence_text,
                ]
            ),
        )
        root_cause = str(analysis.get("root_cause") or "").strip() or str(plan.get("hypothesis") or "已完成调查")
        reply = str(analysis.get("reply") or "").strip() or f"根因判断：{root_cause}"
        confidence = analysis.get("confidence", 0.7)
    else:
        root_cause = str(plan.get("hypothesis") or "当前证据不足")
        reply = f"根因判断：{root_cause}"
        confidence = 0.3

    state["analysis"] = {
        **dict(state.get("analysis") or {}),
        "root_cause": root_cause,
        "reply": reply,
        "investigation_evidence": list(investigation.get("evidence") or []),
    }
    state["root_cause"] = root_cause
    state["solution"] = reply
    state["confidence"] = confidence
    state["analysis_status"] = "SUCCESS"
    state["route"] = "finish"
    return dict(state)


__all__ = ["run"]
