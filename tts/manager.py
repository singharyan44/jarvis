#!/usr/bin/env python3
"""
TTS Manager — handles provider fallback chain.

Iterates through TTS_PREFERENCE_ORDER and uses the first available engine
that successfully speaks the text.
"""

from __future__ import annotations

from core.config import TTS_PREFERENCE_ORDER
from core.logging import log
from tts.elevenlabs_backend import ElevenLabsTTS
from tts.cloud_backends import AzureTTS, GoogleTTS, CoquiTTS
from tts.offline_backend import Pyttsx3TTS
from tts.interface import TTSEngine


# Map preference names to engine classes
_ENGINE_MAP: dict[str, type[TTSEngine]] = {
    "elevenlabs": ElevenLabsTTS,
    "azure": AzureTTS,
    "google": GoogleTTS,
    "coqui": CoquiTTS,
    "pyttsx3": Pyttsx3TTS,
}


class TTSManager:
    """
    Manages TTS provider fallback.

    Usage:
        manager = TTSManager()
        manager.speak("Hello world")
    """

    def __init__(self, preference_order: list[str] | None = None) -> None:
        self.preference_order = preference_order or TTS_PREFERENCE_ORDER
        self._engines: dict[str, TTSEngine] = {}

    def _get_engine(self, name: str) -> TTSEngine | None:
        """Lazy-instantiate engine by name."""
        if name not in self._engines:
            cls = _ENGINE_MAP.get(name)
            if cls is None:
                log.warning("Unknown TTS provider: %s", name)
                return None
            self._engines[name] = cls()
        return self._engines[name]

    def speak(self, text: str) -> bool:
        """
        Speak text using the first available provider in preference order.

        Returns True if any provider succeeded, False if all failed.
        """
        for provider in self.preference_order:
            engine = self._get_engine(provider)
            if engine is None:
                continue
            if not engine.is_available:
                log.debug("%s not available; trying next.", engine.name)
                continue
            if engine.speak(text):
                return True
            log.debug("%s failed; trying next provider.", engine.name)

        log.warning("All TTS providers failed; text not spoken.")
        return False

    def available_providers(self) -> list[str]:
        """Return names of providers that are currently available."""
        return [
            name for name in self.preference_order
            if (eng := self._get_engine(name)) and eng.is_available
        ]