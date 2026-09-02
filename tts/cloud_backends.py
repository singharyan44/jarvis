#!/usr/bin/env python3
"""
Placeholder TTS backends for Azure, Google Cloud, and Coqui.

These return False to trigger fallback to the next provider.
Implement when credentials/keys are available.
"""

from __future__ import annotations

from core.logging import log
from tts.interface import TTSEngine


class AzureTTS(TTSEngine):
    """Placeholder for Azure Cognitive Services TTS."""

    @property
    def name(self) -> str:
        return "Azure TTS"

    @property
    def is_available(self) -> bool:
        # TODO: Check for AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
        return False

    def speak(self, text: str) -> bool:
        log.debug("Azure TTS not configured; skipping.")
        return False


class GoogleTTS(TTSEngine):
    """Placeholder for Google Cloud Text-to-Speech."""

    @property
    def name(self) -> str:
        return "Google Cloud TTS"

    @property
    def is_available(self) -> bool:
        # TODO: Check for GOOGLE_APPLICATION_CREDENTIALS
        return False

    def speak(self, text: str) -> bool:
        log.debug("Google TTS not configured; skipping.")
        return False


class CoquiTTS(TTSEngine):
    """Placeholder for Coqui TTS (offline neural TTS)."""

    @property
    def name(self) -> str:
        return "Coqui TTS"

    @property
    def is_available(self) -> bool:
        # TODO: Check if Coqui is installed and models are available
        return False

    def speak(self, text: str) -> bool:
        log.debug("Coqui TTS not configured; skipping.")
        return False