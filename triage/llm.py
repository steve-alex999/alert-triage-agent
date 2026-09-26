"""Model adapter. The agent keeps its transcript in OpenAI chat format; each provider
translates from that. One OpenAI-compatible adapter covers Gemini and Ollama.

Pick the provider and model with TRIAGE_PROVIDER and TRIAGE_MODEL. TRIAGE_RPM and
TRIAGE_TPM cap requests and tokens per minute; Gemini's free-tier caps differ by model.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

PROVIDERS = {
    # name: (base URL, API-key env var, default model)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
    "ollama": ("http://localhost:11434/v1", None, "qwen3:4b"),
}

# Free-tier caps (requests/min, tokens/min), kept a little under the published limits.
GEMINI_FREE_LIMITS = {
    "gemini-3.5-flash-lite": (14, 240_000),
    "gemini-3.1-flash-lite": (14, 240_000),
    "gemini-3.8-flash": (4, 240_000),
    "gemma-4-26b-a4b-it": (28, 15_000),
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
    wait_s: float = 0.0  # time spent pacing or waiting out rate limits, not the model working


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion: ...


def retry_delay(error: Exception) -> float | None:
    """Seconds a per-minute rate limit asks us to wait, or None for any other 429 (such as a daily quota)."""
    text = str(error)
    if "PerDay" in text:
        return None
    match = re.search(r"retry in ([\d.]+)s", text) or re.search(r"'retryDelay': '(\d+)s'", text)
    return float(match.group(1)) if match else None


class TokenBudget:
    """Keeps the tokens sent in any 60-second window under a per-minute cap, across threads."""

    def __init__(self, tpm: float, clock=time.monotonic, sleep=time.sleep):
        self.tpm = tpm
        self._clock, self._sleep = clock, sleep
        self._spent: list[list[float]] = []  # [time, tokens]
        self._lock = threading.Lock()

    def reserve(self, tokens: float) -> tuple[list[float], float]:
        """Wait until `tokens` fit in the window, then book them. Returns the booking and the wait."""
        tokens = min(tokens, self.tpm)
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._spent = [s for s in self._spent if now - s[0] < 60]
                if sum(s[1] for s in self._spent) + tokens <= self.tpm:
                    booking = [now, tokens]
                    self._spent.append(booking)
                    return booking, waited
                pause = 60 - (now - self._spent[0][0]) + 0.1
            self._sleep(pause)
            waited += pause

    def settle(self, booking: list[float], actual: float) -> None:
        """Replace an estimate with the tokens the provider actually counted."""
        with self._lock:
            booking[1] = actual


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    """Rough upper estimate (3 characters per token) of a request's input tokens."""
    return len(json.dumps(messages)) // 3 + len(json.dumps(tools)) // 3


class OpenAICompatibleLLM:
    def __init__(self, model: str, base_url: str, api_key: str, rpm: float | None = None,
                 tpm: float | None = None, max_retries: int = 6, rate_limit_waits: int = 5):
        from openai import OpenAI

        self.model = model
        # The SDK retries 5xx (free tiers often return 503 under load) with exponential backoff.
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=120)
        self._interval = 60.0 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._slot_lock = threading.Lock()
        self._budget = TokenBudget(tpm) if tpm else None
        self._rate_limit_waits = rate_limit_waits

    def _wait_for_slot(self) -> float:
        """Space requests evenly so a run stays under the per-minute limit across threads."""
        if not self._interval:
            return 0.0
        with self._slot_lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        wait = max(0.0, slot - now)
        time.sleep(wait)
        return wait

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
        from openai import RateLimitError

        waited = 0.0
        booking = None
        if self._budget:
            booking, wait = self._budget.reserve(estimate_tokens(messages, tools))
            waited += wait
        for attempt in range(self._rate_limit_waits + 1):
            waited += self._wait_for_slot()
            try:
                response = self._client.chat.completions.create(model=self.model, messages=messages, tools=tools)
                break
            except RateLimitError as e:
                delay = retry_delay(e)
                if delay is None or attempt == self._rate_limit_waits:
                    raise
                time.sleep(delay + 1)
                waited += delay + 1
        message = response.choices[0].message
        usage = response.usage
        if booking and usage:
            self._budget.settle(booking, usage.total_tokens)
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
            wait_s=waited,
        )


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = provider or os.environ.get("TRIAGE_PROVIDER", "gemini")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown TRIAGE_PROVIDER {provider!r}; use one of {sorted(PROVIDERS)}")
    base_url, key_var, default_model = PROVIDERS[provider]
    api_key = os.environ.get(key_var, "") if key_var else "unused"
    if not api_key:
        raise RuntimeError(f"Set {key_var} to use the {provider} provider")
    model = model or os.environ.get("TRIAGE_MODEL", default_model)
    rpm, tpm = GEMINI_FREE_LIMITS.get(model, (None, None)) if provider == "gemini" else (None, None)
    rpm = float(os.environ.get("TRIAGE_RPM") or 0) or rpm
    tpm = float(os.environ.get("TRIAGE_TPM") or 0) or tpm
    return OpenAICompatibleLLM(model, base_url, api_key, rpm=rpm, tpm=tpm)
