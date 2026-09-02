#!/usr/bin/env python3
"""Audio package — capture, detection, and playback."""

from .capture import play_pcm_f32, play_pcm_i16, record_audio, record_until_silence  # noqa: F401
from .detection import ClapDetector  # noqa: F401
from .wake_word import VoiceWakeDetector, get_voice_wake_detector  # noqa: F401

__all__ = [
    "record_audio",
    "record_until_silence",
    "play_pcm_f32",
    "play_pcm_i16",
    "ClapDetector",
    "VoiceWakeDetector",
    "get_voice_wake_detector",
]