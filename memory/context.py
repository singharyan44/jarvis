#!/usr/bin/env python3
"""
Context Manager (Phase 10) — Intelligent context assembly for LLM calls.

Builds token-limited context packages by:
1. Fetching relevant memory from Honcho
2. Prioritizing: recent messages > representations > search results > summary
3. Applying per-section token budgets
4. Formatting for optimal LLM consumption
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.config import get_vosk_model_path
from core.logging import log
from memory.manager import get_memory_manager


@dataclass
class ContextSection:
    """A single section of assembled context."""
    name: str
    content: str
    tokens: int
    priority: int = 0  # Higher = more important
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPackage:
    """Complete assembled context package for LLM."""
    sections: list[ContextSection] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 4000
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_string(self, separator: str = "\n\n---\n\n") -> str:
        """Format context as string for LLM."""
        parts = []
        for section in self.sections:
            if section.content.strip():
                parts.append(f"[{section.name}]\n{section.content.strip()}")
        return separator.join(parts)
    
    def add_section(self, section: ContextSection) -> bool:
        """Add section if within token budget."""
        if self.total_tokens + section.tokens > self.max_tokens:
            return False
        self.sections.append(section)
        self.total_tokens += section.tokens
        return True


class ContextManager:
    """
    Assembles context packages for LLM calls.
    
    Priority order (highest first):
    1. System prompt / instructions
    2. Recent conversation history
    2. Peer representations (facts/traits)
    3. Relevant search results
    4. Session summary
    """
    
    # Default token budgets per section
    DEFAULT_BUDGETS = {
        "system": 500,
        "recent_history": 1500,
        "representations": 800,
        "search_results": 600,
        "summary": 600,
    }
    
    def __init__(
        self,
        max_tokens: int = 4000,
        budgets: dict[str, int] | None = None,
        peer_id: str = "user",
    ) -> None:
        self.max_tokens = max_tokens
        self.budgets = budgets or self.DEFAULT_BUDGETS.copy()
        self.peer_id = peer_id
        self._memory = get_memory_manager()
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return max(1, len(text) // 4)
    
    async def build_context(
        self,
        user_message: str,
        session_id: str | None = None,
        include_search: bool = True,
        search_query: str | None = None,
    ) -> ContextPackage:
        """
        Build complete context package for an LLM call.
        
        Args:
            user_message: Current user input
            session_id: Current session ID
            include_search: Whether to include search results
            search_query: Query for search (defaults to user_message)
        """
        from memory import get_memory_manager
        memory = get_memory_manager()
        
        pkg = ContextPackage(max_tokens=self.max_tokens)
        pkg.metadata = {
            "peer_id": self.peer_id,
            "session_id": session_id,
            "user_message": user_message,
        }
        
        # 1. System prompt (fixed budget)
        system_prompt = self._get_system_prompt()
        pkg.add_section(ContextSection(
            name="system",
            content=system_prompt,
            tokens=self.estimate_tokens(system_prompt),
            priority=100,
        ))
        
        # 2. Recent history
        if session_id:
            history = await self._get_recent_history(session_id)
            if history:
                pkg.add_section(ContextSection(
                    name="recent_history",
                    content=history,
                    tokens=self.estimate_tokens(history),
                    priority=80,
                    metadata={"session_id": session_id},
                ))
        
        # 3. Peer representations (facts/traits)
        representations = await self._get_representations()
        if representations:
            pkg.add_section(ContextSection(
                name="user_profile",
                content=representations,
                tokens=self.estimate_tokens(representations),
                priority=70,
                metadata={"peer_id": self.peer_id},
            ))
        
        # 4. Search results for current query
        if include_search:
            query = search_query or user_message
            search_results = await self._search(query)
            if search_results:
                pkg.add_section(ContextSection(
                    name="relevant_context",
                    content=search_results,
                    tokens=self.estimate_tokens(search_results),
                    priority=60,
                    metadata={"query": query},
                ))
        
        # 5. Session summary (if available)
        if session_id:
            summary = await self._get_session_summary(session_id)
            if summary:
                pkg.add_section(ContextSection(
                    name="session_summary",
                    content=summary,
                    tokens=self.estimate_tokens(summary),
                    priority=40,
                ))
        
        return pkg
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for JARVIS."""
        return """You are JARVIS, a helpful personal AI assistant.

Capabilities:
- Launch applications (Cursor, Chrome, Spotify)
- Execute system commands
- Answer questions using available tools
- Maintain context across conversations

Guidelines:
- Be concise and helpful
- Use tools when appropriate
- Ask for clarification if ambiguous
- Prioritize user safety and privacy"""
    
    async def _get_recent_history(self, session_id: str) -> str:
        """Get recent messages from session."""
        try:
            from memory import get_memory_manager
            memory = get_memory_manager()
            session = await memory.get_session(session_id)
            if session and session.get("messages"):
                recent = session["messages"][-10:]  # Last 10 messages
                lines = []
                for msg in recent:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    lines.append(f"{role}: {content}")
                return "\n".join(lines)
        except Exception as e:
            log.debug("Failed to get recent history: %s", e)
        return ""
    
    async def _get_representations(self) -> str:
        """Get peer representations (facts/traits)."""
        try:
            from memory import get_memory_manager
            memory = get_memory_manager()
            reps = await memory.get_peer_representations(self.peer_id)
            if not reps:
                return ""
            
            lines = ["Known facts about user:"]
            for rep in reps[:20]:  # Limit to top 20
                content = rep.get("content", "")
                if content:
                    lines.append(f"- {content}")
            return "\n".join(lines)
        except Exception as e:
            log.debug("Failed to get representations: %s", e)
        return ""
    
    async def _search(self, query: str) -> str:
        """Search for relevant context."""
        try:
            from memory import get_memory_manager
            memory = get_memory_manager()
            results = await memory.search(query, peer_id=self.peer_id, limit=5)
            if not results:
                return ""
            
            lines = [f"Relevant context for: {query}"]
            for r in results:
                content = r.get("content") or r.get("text") or str(r)
                if content:
                    lines.append(f"- {content[:200]}")
            return "\n".join(lines)
        except Exception as e:
            log.debug("Search failed: %s", e)
        return ""
    
    async def _get_session_summary(self, session_id: str) -> str:
        """Get session summary if available."""
        try:
            from memory import get_memory_manager
            memory = get_memory_manager()
            summary = await memory.get_session_summary(session_id)
            if summary:
                return f"Session summary: {summary}"
        except Exception as e:
            log.debug("Failed to get session summary: %s", e)
        return ""


# Global instance
_context_manager = None


def get_context_manager(
    max_tokens: int = 4000,
    peer_id: str = "user",
) -> ContextManager:
    """Get or create global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(max_tokens=max_tokens, peer_id=peer_id)
    return _context_manager