"""Resilient LLM interface with retries, timeouts, and circuit breaker.

Features:
- Exponential backoff retries (3 attempts)
- Request timeouts (30 seconds)
- Circuit breaker (fail after 5 consecutive failures)
- Multi-LLM fallback (Claude → OpenAI → Groq → Ollama)
- Cost tracking and metrics
- Token counting
"""

import time
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

logger = logging.getLogger("llm_resilient")


class LLMCircuitBreaker:
    """Simple circuit breaker for LLM calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                "circuit_breaker_open",
                extra={"failures": self.failure_count}
            )

    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        if not self.is_open:
            return True
        
        # Check if recovery timeout has elapsed
        if time.time() - self.last_failure_time > self.recovery_timeout:
            logger.info("circuit_breaker_attempting_recovery")
            self.is_open = False
            self.failure_count = 0
            return True
        
        return False


# Circuit breakers per LLM provider
_breakers = {
    "anthropic": LLMCircuitBreaker(),
    "openai": LLMCircuitBreaker(),
    "groq": LLMCircuitBreaker(),
    "ollama": LLMCircuitBreaker(),
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def call_llm_with_fallback(
    prompt: str,
    system: str = "You are an expert DevOps AI assistant.",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """
    Call LLM with automatic retry and fallback chain.
    
    Tries: Claude → OpenAI → Groq → Ollama (local)
    
    Returns:
        Generated text response
        
    Raises:
        TimeoutError: If all LLMs timeout
        Exception: If all LLMs fail
    """
    # from app.core.metrics import (
    #     llm_calls_total,
    #     llm_request_duration_seconds,
    #     external_api_failures_total,
    # )
    from app.core.exceptions import CircuitBreakerOpen, TimeoutError as APITimeout
    
    providers = ["anthropic", "openai", "groq", "ollama"]
    last_error = None
    
    for provider in providers:
        # Check circuit breaker
        breaker = _breakers[provider]
        if not breaker.is_available():
            logger.warning(f"circuit_breaker_open for {provider}")
            # llm_calls_total.labels(
            #     provider=provider,
            #     model="unknown",
            #     status="circuit_breaker_open"
            # ).inc()
            continue
        
        try:
            start = time.time()
            
            # Call the provider-specific implementation
            if provider == "anthropic":
                result = _call_anthropic(prompt, system, max_tokens, temperature)
            elif provider == "openai":
                result = _call_openai(prompt, system, max_tokens, temperature)
            elif provider == "groq":
                result = _call_groq(prompt, system, max_tokens, temperature)
            elif provider == "ollama":
                result = _call_ollama(prompt, system, max_tokens, temperature)
            
            duration = time.time() - start
            
            # Success! Record metrics
            # llm_calls_total.labels(
            #     provider=provider,
            #     model=result.get("model", "unknown"),
            #     status="success"
            # ).inc()
            
            # llm_request_duration_seconds.labels(
            #     provider=provider,
            #     model=result.get("model", "unknown")
            # ).observe(duration)
            
            breaker.record_success()
            logger.info(f"llm_success provider={provider} duration={duration:.2f}s tokens={result.get('tokens', 0)}")
            
            return result["content"]
        
        except asyncio.TimeoutError:
            duration = time.time() - start
            logger.warning(f"llm_timeout provider={provider} after {duration:.1f}s")
            # llm_calls_total.labels(
            #     provider=provider,
            #     model="unknown",
            #     status="timeout"
            # ).inc()
            # external_api_failures_total.labels(
            #     service="llm",
            #     reason="timeout"
            # ).inc()
            breaker.record_failure()
            last_error = APITimeout(service=provider, timeout_sec=30)
            continue
        
        except Exception as e:
            logger.warning(f"llm_error provider={provider} error={str(e)}")
            # llm_calls_total.labels(
            #     provider=provider,
            #     model="unknown",
            #     status="error"
            # ).inc()
            # external_api_failures_total.labels(
            #     service="llm",
            #     reason=type(e).__name__
            # ).inc()
            breaker.record_failure()
            last_error = e
            continue
    
    # All providers failed
    logger.error("all_llm_providers_failed")
    raise last_error or Exception("All LLM providers unavailable")


def _call_anthropic(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call Anthropic Claude with timeout."""
    from app.llm.claude import _anthropic_client
    # from app.core.metrics import llm_tokens_total, llm_cost_usd
    
    if not _anthropic_client:
        raise ValueError("Anthropic client not configured")
    
    # Call with timeout
    try:
        response = _anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,  # SRE: explicit timeout
        )
    except Exception as e:
        if "timeout" in str(e).lower():
            raise asyncio.TimeoutError(f"Anthropic timeout: {e}")
        raise
    
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens
    
    # Track tokens
    # llm_tokens_total.labels(
    #     provider="anthropic",
    #     model="claude-3-sonnet"
    # ).inc(total_tokens)
    
    # Track cost (Claude pricing: ~$0.003/$0.015 per 1K tokens)
    # cost = (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.015
    # llm_cost_usd.labels(
    #     provider="anthropic",
    #     model="claude-3-sonnet"
    # ).inc(cost)
    
    return {
        "content": response.content[0].text,
        "model": "claude-3-sonnet",
        "provider": "anthropic",
        "tokens": total_tokens,
        "cost": cost,
    }


def _call_openai(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call OpenAI GPT-4 with timeout."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ValueError("OpenAI not installed")
    
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key, timeout=30.0)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
        )
    except Exception as e:
        if "timeout" in str(e).lower():
            raise asyncio.TimeoutError(f"OpenAI timeout: {e}")
        raise
    
    total_tokens = response.usage.total_tokens
    
    # from app.core.metrics import llm_tokens_total, llm_cost_usd
    # llm_tokens_total.labels(
    #     provider="openai",
    #     model="gpt-4-turbo"
    # ).inc(total_tokens)
    
    # OpenAI pricing: ~$0.03/$0.06 per 1K tokens for GPT-4
    # cost = (response.usage.prompt_tokens / 1000) * 0.03 + (response.usage.completion_tokens / 1000) * 0.06
    # llm_cost_usd.labels(
    #     provider="openai",
    #     model="gpt-4-turbo"
    # ).inc(cost)
    
    return {
        "content": response.choices[0].message.content,
        "model": "gpt-4-turbo",
        "provider": "openai",
        "tokens": total_tokens,
        "cost": cost,
    }


def _call_groq(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call Groq Mixtral with timeout."""
    try:
        from groq import Groq
    except ImportError:
        raise ValueError("Groq not installed")
    
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    client = Groq(api_key=api_key, timeout=30.0)
    
    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
        )
    except Exception as e:
        if "timeout" in str(e).lower():
            raise asyncio.TimeoutError(f"Groq timeout: {e}")
        raise
    
    total_tokens = response.usage.total_tokens
    
    # from app.core.metrics import llm_tokens_total, llm_cost_usd
    # llm_tokens_total.labels(
    #     provider="groq",
    #     model="mixtral-8x7b"
    # ).inc(total_tokens)
    
    # Groq pricing: very cheap, ~$0.27 per 1M tokens
    # cost = (total_tokens / 1_000_000) * 0.27
    # llm_cost_usd.labels(
    #     provider="groq",
    #     model="mixtral-8x7b"
    # ).inc(cost)
    
    return {
        "content": response.choices[0].message.content,
        "model": "mixtral-8x7b",
        "provider": "groq",
        "tokens": total_tokens,
        "cost": cost,
    }


def _call_ollama(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call local Ollama with timeout."""
    import os
    import requests
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "llama2",
                "prompt": f"{system}\n\n{prompt}",
                "stream": False,
                "temperature": temperature,
            },
            timeout=30.0,  # SRE: explicit timeout
        )
        response.raise_for_status()
    except requests.Timeout:
        raise asyncio.TimeoutError(f"Ollama timeout after 30s")
    except Exception as e:
        raise ValueError(f"Ollama error: {e}")
    
    data = response.json()
    
    return {
        "content": data.get("response", ""),
        "model": "llama2",
        "provider": "ollama",
        "tokens": 0,  # Ollama doesn't provide token counts
        "cost": 0,    # Local, no cost
    }
