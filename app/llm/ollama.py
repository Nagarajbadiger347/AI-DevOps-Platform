"""Ollama LLM provider — runs models locally with zero API cost.

Supports any model pulled via `ollama pull`, including:
  - llama3, llama3:8b, llama3:70b
  - mistral, mistral-nemo
  - gemma2, phi3
  - Your fine-tuned NexusOps model (after training)

Config (env vars):
  OLLAMA_BASE_URL  — default: http://localhost:11434
  OLLAMA_MODEL     — default: llama3
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from app.llm.base import BaseLLM, LLMResponse

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


class OllamaProvider(BaseLLM):
    """Local Ollama inference — OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, base_url: str = _BASE_URL, model: str = _MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model

    def is_available(self) -> bool:
        """Ping the Ollama API to check if it's running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        *,
        system: str = "You are an expert DevOps AI assistant.",
        max_tokens: int = 2048,
        messages: Optional[list] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Build message list
        msg_list = [{"role": "system", "content": system}]
        if messages:
            msg_list.extend(messages)
        if prompt:
            msg_list.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":       self.model,
            "messages":    msg_list,
            "stream":      False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data    = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                usage   = data.get("usage", {})
                logger.info(
                    "ollama_completion",
                    extra={
                        "model": self.model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                )
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="ollama",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from e
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Ollama response parse error: {e}") from e
