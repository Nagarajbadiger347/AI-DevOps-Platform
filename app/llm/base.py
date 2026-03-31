"""BaseLLM interface — all providers must implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    ) -> LLMResponse:
        """Synchronous completion — returns a structured LLMResponse."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""
        ...
