#!/usr/bin/env python3
"""STT package — interface and backends."""

from .interface import STTEngine  # noqa: F401
from .vosk_backend import VoskSTT  # noqa: F401
from .whisper_backend import WhisperModelManager, WhisperSTT  # noqa: F401

__all__ = [
    "STTEngine",
    "WhisperSTT",
    "WhisperModelManager",
    "VoskSTT",
]