"""Model adapter. The agent keeps its transcript in OpenAI chat format; each provider
translates from that. One OpenAI-compatible adapter covers Gemini and Ollama.

Pick the provider and model with TRIAGE_PROVIDER and TRIAGE_MODEL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

PROVIDERS = {
    # name: (base URL, API-key env var, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "ollama": ("http://localhost:11434/v1", None, "qwen3:4b"),
}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    message: dict[str, Any] = field(repr=False)  # appended to the transcript as-is
    text: str | None
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion: ...


class OpenAICompatibleLLM:
    def __init__(self, model: str, base_url: str, api_key: str, max_retries: int = 6):
        from openai import OpenAI

        self.model = model
        # The SDK retries 429s and 5xx with exponential backoff; free tiers need a few extra tries.
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=120)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        response = self._client.chat.completions.create(model=self.model, messages=messages, tools=tools)
        message = response.choices[0].message
        usage = response.usage
        return Completion(
            # Keep provider extras (Gemini's thought signatures must be sent back on the next turn).
            message=message.model_dump(exclude_none=True),
            text=message.content,
            tool_calls=[
                ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
                for c in message.tool_calls or []
            ],
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = provider or os.environ.get("TRIAGE_PROVIDER", "gemini")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown TRIAGE_PROVIDER {provider!r}; use one of {sorted(PROVIDERS)}")
    base_url, key_var, default_model = PROVIDERS[provider]
    api_key = os.environ.get(key_var, "") if key_var else "unused"
    if not api_key:
        raise RuntimeError(f"Set {key_var} to use the {provider} provider")
    return OpenAICompatibleLLM(model or os.environ.get("TRIAGE_MODEL", default_model), base_url, api_key)
