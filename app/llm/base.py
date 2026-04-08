"""BaseLLM interface — all providers must implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLM(ABC):
    """Minimal interface every LLM provider must implement."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "You are an expert DevOps AI assistant.",
        max_tokens: int = 2048,
        messages: Optional[list] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Synchronous completion — returns a structured LLMResponse.

        Args:
            prompt:      Final user turn text (appended after messages if provided).
            system:      System prompt.
            max_tokens:  Max tokens to generate.
            messages:    Optional list of prior turns [{"role": "user"|"assistant", "content": str}].
                         When provided, prompt is appended as the final user turn.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""
        ...
