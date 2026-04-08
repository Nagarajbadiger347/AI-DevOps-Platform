"""OpenAI LLM provider — implements BaseLLM interface."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.llm.base import BaseLLM, LLMResponse

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

_client = None
if _OPENAI_API_KEY:
    try:
        from openai import OpenAI as _OpenAI
        _client = _OpenAI(api_key=_OPENAI_API_KEY)
    except ImportError:
        pass


class OpenAIProvider(BaseLLM):
    """OpenAI GPT provider — used as fallback when Claude is unavailable."""

    def is_available(self) -> bool:
        return _client is not None

    def complete(
        self,
        prompt: str,
        *,
        system: str = "You are an expert DevOps AI assistant.",
        max_tokens: int = 2048,
        messages: list | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if _client is None:
            raise RuntimeError(
                "OpenAI provider unavailable — add OPENAI_API_KEY to .env"
            )
        # Build structured message list with full history
        msg_list = [{"role": "system", "content": system}]
        if messages:
            msg_list.extend(messages)
        msg_list.append({"role": "user", "content": prompt})
        resp = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msg_list,
        )
        choice = resp.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            provider="openai",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
