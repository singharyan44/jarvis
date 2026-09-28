#!/usr/bin/env python3
"""
ElevenLabs TTS backend.

High-quality cloud TTS with caching support.
Uses the modern ElevenLabs SDK (v1+) with client.text_to_speech.convert().
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from core.config import (
    JARVIS_WELCOME_CACHE_ENABLED,
    elevenlabs_env_config,
    jarvis_welcome_cache_dir,
    jarvis_welcome_cache_path,
)
from core.logging import log
from tts.interface import TTSEngine


@contextlib.contextmanager
def _without_sslkeylogfile():
    """Temporarily disable inherited SSL key logging that can break sandboxed HTTP clients."""
    prev = os.environ.pop("SSLKEYLOGFILE", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["SSLKEYLOGFILE"] = prev


def _elevenlabs_client_cls():
    """Get the ElevenLabs client class, handling import variations."""
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            return None
    return ElevenLabs


def _elevenlabs_audio_bytes(
    text: str, voice_id: str, model_id: str, output_format: str, api_key: str
) -> bytes:
    """Fetch audio bytes from ElevenLabs API using the modern client."""
    client_cls = _elevenlabs_client_cls()
    if client_cls is None:
        raise ImportError("ElevenLabs SDK is not installed.")
    with _without_sslkeylogfile():
        client = client_cls(api_key=api_key)
        return b"".join(
            client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
            )
        )


def _save_pcm_wav_file(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    """Atomically save PCM bytes as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        tmp.replace(path)
    except (OSError, wave.Error) as e:
        log.warning("Could not write cached welcome audio: %s", e)
        if tmp.exists():
            tmp.unlink()


def _play_pcm_wav_file(path: Path) -> bool:
    """Play a cached PCM WAV file."""
    try:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            if ch != 1 or sw != 2:
                log.warning("Unsupported cached WAV (channels=%s, width=%s).", ch, sw)
                return False
            raw = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as e:
        log.warning("Could not read cached welcome audio: %s", e)
        return False
    if not raw:
        return False
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f = pcm_i16.astype(np.float32) / 32768.0
    try:
        sd.play(pcm_f, rate)
        sd.wait()
    except Exception as e:
        log.warning("Could not play cached welcome audio: %s", e)
        return False
    return True


class ElevenLabsTTS(TTSEngine):
    """ElevenLabs cloud TTS with WAV caching."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        self._voice_id, self._model_id, self._output_format, self._pcm_sr = elevenlabs_env_config()

    @property
    def name(self) -> str:
        return "ElevenLabs"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self._voice_id)

    def speak(self, text: str) -> bool:
        if not self.is_available:
            if not self._api_key:
                log.warning("ELEVENLABS_API_KEY not set; skipping ElevenLabs.")
            elif not self._voice_id:
                log.warning("ELEVENLABS_VOICE_ID not set; skipping ElevenLabs.")
            return False

        cache_path = jarvis_welcome_cache_path(text, self._voice_id, self._model_id, self._output_format)

        # Try cache first
        if JARVIS_WELCOME_CACHE_ENABLED and cache_path.exists():
            if _play_pcm_wav_file(cache_path):
                log.info("Played cached ElevenLabs audio for: %s", text[:50])
                return True

        # Fetch from API
        try:
            audio_bytes = _elevenlabs_audio_bytes(
                text, self._voice_id, self._model_id, self._output_format, self._api_key
            )
            if not audio_bytes:
                log.warning("ElevenLabs returned empty audio for: %s", text[:50])
                return False

            # Convert to PCM if needed
            if self._output_format.startswith("pcm_"):
                pcm_bytes = audio_bytes
            else:
                # Convert MP3/OGG/etc to PCM using pydub
                try:
                    from pydub import AudioSegment
                except ImportError:
                    log.warning("pydub not installed; cannot convert %s to PCM.", self._output_format)
                    return False

                raw = io.BytesIO(audio_bytes)
                fmt = self._output_format.split("_", 1)[0]
                seg = AudioSegment.from_file(raw, format=fmt)
                pcm_bytes = seg.set_frame_rate(self._pcm_sr).raw_data

            # Cache if enabled
            if JARVIS_WELCOME_CACHE_ENABLED:
                _save_pcm_wav_file(cache_path, pcm_bytes, self._pcm_sr)

            # Play
            sd.play(np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0, self._pcm_sr)
            sd.wait()
            log.info("Spoke via ElevenLabs: %s", text[:50])
            return True

        except Exception as e:
            log.warning("ElevenLabs TTS failed: %s", e)
            return False