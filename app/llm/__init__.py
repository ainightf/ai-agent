"""
LLM 模块 - 大模型封装
"""
from app.llm.llm import LLM
from app.llm.prompt import (
    AGENT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RAG_QUERY_PROMPT,
    get_agent_prompt,
    get_planner_prompt,
    get_rag_prompt
)

__all__ = [
    "LLM",
    "AGENT_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT", 
    "RAG_QUERY_PROMPT",
    "get_agent_prompt",
    "get_planner_prompt",
    "get_rag_prompt"
]
