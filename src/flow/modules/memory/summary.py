"""Summary helpers for memory module."""

from __future__ import annotations


def summarize_with_llm_placeholder(total_message: str, summary_message: str) -> str:
    """
    占位：后续接入真实大模型总结能力。
    当前先把完整消息 + 历史总结做拼接压缩。
    """
    total_text = str(total_message).strip()
    summary_text = str(summary_message).strip()
    if not total_text and not summary_text:
        return ""
    merged = f"{summary_text}\n{total_text}".strip()
    return merged[:500]

