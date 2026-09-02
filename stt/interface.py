#!/usr/bin/env python3
"""
STT (Speech-to-Text) interface.

Defines the abstract base class for STT backends.
Concrete implementations (Whisper, Vosk, etc.) should subclass this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class STTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to a WAV file (mono, 16-bit PCM).

        Returns:
            Transcribed text, empty string if silence or error.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the engine is ready to use (model loaded, etc.)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the engine."""
        ...