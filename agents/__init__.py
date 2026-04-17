"""Agent package for the ScoreG local LLM experiment."""

from .ollama_agent import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OFFICIAL_OLLAMA_DOWNLOAD_URL,
    AgentDecision,
    AgentExecutionError,
    OllamaAgent,
    OllamaStatus,
    build_agent_user_prompt,
    check_ollama_setup,
)

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "OFFICIAL_OLLAMA_DOWNLOAD_URL",
    "AgentDecision",
    "AgentExecutionError",
    "OllamaAgent",
    "OllamaStatus",
    "build_agent_user_prompt",
    "check_ollama_setup",
]
