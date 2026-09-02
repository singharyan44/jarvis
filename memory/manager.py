#!/usr/bin/env python3
"""
Memory Manager — Honcho integration for persistent memory (Phase 9).

Provides high-level API for conversation storage, search, and retrieval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from core.config import (
    HONCHO_API_URL,
    HONCHO_WORKSPACE,
    HONCHO_ENABLED,
)
from core.logging import log


@dataclass
class MemoryConfig:
    """Honcho memory configuration."""
    api_url: str = HONCHO_API_URL
    workspace: str = HONCHO_WORKSPACE
    enabled: bool = HONCHO_ENABLED
    timeout: float = 10.0


class MemoryManager:
    """
    High-level interface to Honcho memory system.
    
    Handles:
    - Workspace/peer/session management
    - Message storage with automatic reasoning
    - Hybrid search (semantic + keyword)
    - Context retrieval for LLM calls
    - Peer representations/facts
    - Auto-summarization of sessions
    """
    
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # ---- Workspace Management ----
    
    async def ensure_workspace(self) -> bool:
        """Create workspace if it doesn't exist."""
        if not self.config.enabled:
            return False
        client = await self._get_client()
        try:
            resp = await client.post(
                "/workspaces",
                json={"name": self.config.workspace},
            )
            return resp.status_code in (200, 201, 409)  # 409 = already exists
        except Exception as e:
            log.error("Failed to ensure workspace: %s", e)
            return False
    
    # ---- Peer Management ----
    
    async def get_or_create_peer(self, peer_id: str, metadata: dict | None = None) -> dict:
        """Get or create a peer (user/agent)."""
        if not self.config.enabled:
            return {}
        client = await self._get_client()
        try:
            # Try to get existing
            resp = await client.get(f"/peers/{peer_id}")
            if resp.status_code == 200:
                return resp.json()
            
            # Create new
            resp = await client.post(
                "/peers",
                json={"id": peer_id, "metadata": metadata or {}},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error("Failed to get/create peer: %s", e)
            return {}
    
    # ---- Session Management ----
    
    async def create_session(self, peer_id: str, metadata: dict | None = None) -> str | None:
        """Create a new session for a peer."""
        if not self.config.enabled:
            return None
        client = await self._get_client()
        try:
            resp = await client.post(
                "/sessions",
                json={
                    "workspace": self.config.workspace,
                    "peer_id": peer_id,
                    "metadata": metadata or {},
                },
            )
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            log.error("Failed to create session: %s", e)
            return None
    
    async def get_session(self, session_id: str) -> dict | None:
        """Get session details."""
        if not self.config.enabled:
            return None
        client = await self._get_client()
        try:
            resp = await client.get(f"/sessions/{session_id}")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            log.error("Failed to get session: %s", e)
        return None
    
    async def get_session_summary(self, session_id: str) -> str | None:
        """Get auto-generated session summary from Honcho."""
        if not self.config.enabled:
            return None
        client = await self._get_client()
        try:
            # Honcho sessions have summary field
            resp = await client.get(f"/sessions/{session_id}")
            if resp.status_code == 200:
                session = resp.json()
                return session.get("summary")
        except Exception as e:
            log.error("Failed to get session summary: %s", e)
        return None
    
    # ---- Message Storage ----
    
    async def add_message(
        self,
        session_id: str,
        role: str,  # "user" or "assistant"
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """Store a message in the session."""
        if not self.config.enabled:
            return False
        client = await self._get_client()
        try:
            resp = await client.post(
                f"/sessions/{session_id}/messages",
                json={
                    "role": role,
                    "content": content,
                    "metadata": metadata or {},
                },
            )
            return resp.status_code in (200, 201)
        except Exception as e:
            log.error("Failed to add message: %s", e)
            return False
    
    async def log_interaction(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        intent: dict | None = None,
    ) -> bool:
        """Log a complete interaction turn (user + assistant)."""
        ok = True
        ok &= await self.add_message(session_id, "user", user_input, {"intent": intent})
        ok &= await self.add_message(session_id, "assistant", assistant_response)
        return ok
    
    # ---- Context Retrieval ----
    
    async def get_context(
        self,
        peer_id: str,
        max_tokens: int = 2000,
        session_id: str | None = None,
    ) -> dict | None:
        """Get token-limited context for a peer (includes auto-summary)."""
        if not self.config.enabled:
            return None
        client = await self._get_client()
        try:
            params = {"peer_id": peer_id, "max_tokens": max_tokens}
            if session_id:
                params["session_id"] = session_id
            
            resp = await client.get("/context", params=params)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            log.error("Failed to get context: %s", e)
        return None
    
    # ---- Search ----
    
    async def search(
        self,
        query: str,
        peer_id: str | None = None,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[dict]:
        """Hybrid search across messages and memories."""
        if not self.config.enabled:
            return []
        client = await self._get_client()
        try:
            payload = {
                "query": query,
                "limit": limit,
            }
            if peer_id:
                payload["peer_id"] = peer_id
            if session_id:
                payload["session_id"] = session_id
            
            resp = await client.post("/search", json=payload)
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except Exception as e:
            log.error("Search failed: %s", e)
        return []
    
    # ---- Peer Representations (Facts) ----
    
    async def get_peer_representations(self, peer_id: str) -> list[dict]:
        """Get extracted facts/representations about a peer (auto-summarized)."""
        if not self.config.enabled:
            return []
        client = await self._get_client()
        try:
            resp = await client.get(f"/peers/{peer_id}/representations")
            if resp.status_code == 200:
                return resp.json().get("representations", [])
        except Exception as e:
            log.error("Failed to get representations: %s", e)
        return []
    
    # ---- Chat/Dialectic ----
    
    async def chat_with_peer(
        self,
        peer_id: str,
        message: str,
        session_id: str | None = None,
    ) -> str | None:
        """Natural language query about a peer (Dialectic API)."""
        if not self.config.enabled:
            return None
        client = await self._get_client()
        try:
            payload = {"peer_id": peer_id, "message": message}
            if session_id:
                payload["session_id"] = session_id
            
            resp = await client.post("/peers/chat", json=payload)
            if resp.status_code == 200:
                return resp.json().get("response")
        except Exception as e:
            log.error("Peer chat failed: %s", e)
        return None


# Global instance
_memory_manager: MemoryManager | None = None


def get_memory_manager(config: MemoryConfig | None = None) -> MemoryManager:
    """Get or create global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(config)
    return _memory_manager


async def close_memory_manager() -> None:
    """Close global memory manager."""
    global _memory_manager
    if _memory_manager is not None:
        await _memory_manager.close()
        _memory_manager = None