#!/usr/bin/env python3
"""
TTS Engine interface.

All TTS backends should implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine is currently available (keys installed, etc.)."""
        ...

    @abstractmethod
    def speak(self, text: str) -> bool:
        """
        Speak the given text.

        Returns True on success, False to trigger fallback.
        """
        ...