#!/usr/bin/env python3
"""TTS package — interface, manager, and backends."""

from .cloud_backends import AzureTTS, CoquiTTS, GoogleTTS  # noqa: F401
from .elevenlabs_backend import ElevenLabsTTS  # noqa: F401
from .interface import TTSEngine  # noqa: F401
from .manager import TTSManager  # noqa: F401
from .offline_backend import Pyttsx3TTS  # noqa: F401

__all__ = [
    "TTSEngine",
    "TTSManager",
    "ElevenLabsTTS",
    "AzureTTS",
    "GoogleTTS",
    "CoquiTTS",
    "Pyttsx3TTS",
]