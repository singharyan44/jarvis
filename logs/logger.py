#!/usr/bin/env python3
"""
Session Logger — JSONL structured logging for JARVIS.

Writes session events to logs/raw_session.jsonl with the format:
{
  "time": "2026-07-06T12:34:56.789Z",
  "type": "command|tts|error|ai_call|performance",
  "input": "...",
  "action": "...",
  "latency": 0.4,
  "success": true,
  "details": {...}
}

Entries are appended line-by-line for easy parsing and analysis.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Compute BASE_DIR locally to avoid circular imports
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class SessionEvent:
    """Structured session event for JSONL logging."""
    type: str  # "command", "tts", "error", "ai_call", "performance", "startup", "shutdown"
    input: str = ""
    action: str = ""
    latency: float = 0.0
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"), ensure_ascii=False)


class SessionLogger:
    """
    Thread-safe JSONL session logger.

    Usage:
        logger = SessionLogger()
        logger.log_command("open cursor", "open_cursor", "launch", 0.15, True)
        logger.log_tts("elevenlabs", "Welcome home sir.", 0.8, True)
        logger.log_error("Whisper transcription failed", "cuda_out_of_memory")
    """

    def __init__(self, log_dir: Path | str | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else (BASE_DIR / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "raw_session.jsonl"
        self._lock = threading.Lock()
        self._session_start = time.time()

        # Log session start
        self._write(SessionEvent(
            type="startup",
            action="session_start",
            success=True,
            details={"pid": __import__("os").getpid()}
        ))

    def _write(self, event: SessionEvent) -> None:
        """Thread-safe write to JSONL file."""
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                # Don't let logging errors crash the app
                import logging
                logging.getLogger("jarvis").warning("Failed to write session log: %s", e)

    def log_command(
        self,
        transcript: str,
        tool: str,
        action: str,
        latency: float,
        success: bool,
        message: str = "",
        parameters: dict | None = None,
    ) -> None:
        """Log a command execution event."""
        self._write(SessionEvent(
            type="command",
            input=transcript,
            action=f"{tool}.{action}",
            latency=latency,
            success=success,
            details={
                "tool": tool,
                "action": action,
                "message": message,
                "parameters": parameters or {},
            }
        ))

    def log_tts(
        self,
        provider: str,
        text: str,
        latency: float,
        success: bool,
        cached: bool = False,
        error: str = "",
    ) -> None:
        """Log a TTS usage event."""
        self._write(SessionEvent(
            type="tts",
            input=text[:200],  # Truncate long text
            action=f"speak:{provider}",
            latency=latency,
            success=success,
            details={
                "provider": provider,
                "cached": cached,
                "error": error,
                "text_length": len(text),
            }
        ))

    def log_error(
        self,
        error_message: str,
        error_type: str = "exception",
        context: dict | None = None,
    ) -> None:
        """Log an error/failure event."""
        self._write(SessionEvent(
            type="error",
            input=error_message,
            action=error_type,
            success=False,
            details=context or {},
        ))

    def log_ai_call(
        self,
        prompt: str,
        model: str,
        latency: float,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str = "",
    ) -> None:
        """Log an AI/LLM API call event."""
        self._write(SessionEvent(
            type="ai_call",
            input=prompt[:500],
            action=f"llm:{model}",
            latency=latency,
            success=success,
            details={
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "error": error,
            }
        ))

    def log_performance(
        self,
        operation: str,
        latency: float,
        success: bool,
        metadata: dict | None = None,
    ) -> None:
        """Log a performance measurement."""
        self._write(SessionEvent(
            type="performance",
            action=operation,
            latency=latency,
            success=success,
            details=metadata or {},
        ))

    def log_shutdown(self, reason: str = "user_interrupt") -> None:
        """Log session end."""
        uptime = time.time() - self._session_start
        self._write(SessionEvent(
            type="shutdown",
            action=reason,
            latency=uptime,
            success=True,
            details={"uptime_seconds": uptime}
        ))

    def get_log_path(self) -> Path:
        """Return the path to the session log file."""
        return self.log_path


# Global instance for easy access
_session_logger: SessionLogger | None = None


def get_session_logger() -> SessionLogger:
    """Get or create the global session logger instance."""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger()
    return _session_logger


def close_session_logger(reason: str = "normal") -> None:
    """Close the global session logger and log shutdown."""
    global _session_logger
    if _session_logger is not None:
        _session_logger.log_shutdown(reason)
        _session_logger = None