"""Summary helpers for memory module."""

from __future__ import annotations

from llm.llm import summarize_with_llm


def summarize_with_llm_placeholder(total_message: str, summary_message: str) -> str:
    """
    兼容旧方法名：实际调用统一收口至 llm.llm.summarize_with_llm。
    """
    return summarize_with_llm(total_message=total_message, summary_message=summary_message)
