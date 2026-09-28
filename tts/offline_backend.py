#!/usr/bin/env python3
"""
pyttsx3 TTS backend.

Guaranteed offline fallback using system TTS engine.
"""

from __future__ import annotations

import pyttsx3

from core.logging import log
from tts.interface import TTSEngine


class Pyttsx3TTS(TTSEngine):
    """Offline TTS using pyttsx3 (system voices)."""

    def __init__(self) -> None:
        self._engine = None

    @property
    def name(self) -> str:
        return "pyttsx3 (offline)"

    @property
    def is_available(self) -> bool:
        return True  # Always available as last resort

    def speak(self, text: str) -> bool:
        try:
            if self._engine is None:
                self._engine = pyttsx3.init()
            self._engine.say(text)
            self._engine.runAndWait()
            log.info("Spoke via pyttsx3: %s", text[:50])
            return True
        except Exception as e:
            log.warning("pyttsx3 TTS failed: %s", e)
            return False