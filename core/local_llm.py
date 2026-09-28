#!/usr/bin/env python3
"""
Local LLM support for offline mode (Phase 8).

Supports Ollama and vLLM compatible OpenAI endpoints.
"""

from __future__ import annotations

import os
import httpx
from dataclasses import dataclass
from typing import Any

from core.config import BASE_DIR
from core.logging import log


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM provider."""
    provider: str  # "ollama", "vllm", "lmstudio"
    base_url: str
    model: str
    timeout: float = 30.0


# Default local LLM configurations
LOCAL_LLM_CONFIGS = {
    "ollama": LocalLLMConfig(
        provider="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
    ),
    "vllm": LocalLLMConfig(
        provider="vllm",
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
    ),
    "lmstudio": LocalLLMConfig(
        provider="lmstudio",
        base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        model=os.environ.get("LMSTUDIO_MODEL", "local-model"),
    ),
}


class LocalLLMClient:
    """
    Client for local LLM inference via OpenAI-compatible API.
    
    Supports Ollama, vLLM, LM Studio, and any OpenAI-compatible endpoint.
    """
    
    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LOCAL_LLM_CONFIGS["ollama"]
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 500,
        **kwargs
    ) -> dict[str, Any] | None:
        """Call local LLM chat completion endpoint."""
        client = await self._get_client()
        try:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error("Local LLM call failed (%s): %s", self.config.provider, e)
            return None
    
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def is_available(self) -> bool:
        """Check if local LLM server is reachable (sync check)."""
        import requests
        try:
            r = requests.get(f"{self.config.base_url}/models", timeout=2)
            return r.status_code == 200
        except Exception:
            return False


def get_local_llm_client(provider: str = "ollama") -> LocalLLMClient | None:
    """Factory to get local LLM client."""
    config = LOCAL_LLM_CONFIGS.get(provider)
    if not config:
        return None
    client = LocalLLMClient(config)
    if client.is_available():
        log.info("Local LLM available: %s at %s", provider, config.base_url)
        return client
    log.debug("Local LLM not available: %s at %s", provider, config.base_url)
    return None


def get_available_local_llms() -> list[str]:
    """Return list of available local LLM providers."""
    available = []
    for name in LOCAL_LLM_CONFIGS:
        if get_local_llm_client(name):
            available.append(name)
    return available