#!/usr/bin/env python3
"""
Vosk STT backend — fully offline speech recognition.

Supports dual models:
- Small model (~40MB): wake word + basic STT fallback
- Large model (~1.8GB): high-accuracy offline STT
- LGraph variant (~50MB): compressed large model
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.config import (
    TRANSCRIPTION_SILENCE_RMS,
    VOSK_MODEL_PATH,
    VOSK_MODEL_SIZE,
    VOSK_AUTO_DOWNLOAD,
    VOSK_SAMPLE_RATE,
    get_vosk_model_path,
)
from core.logging import log
from stt.interface import STTEngine

try:
    import vosk  # type: ignore
except ImportError:
    vosk = None  # type: ignore


class VoskSTT(STTEngine):
    """Vosk offline STT engine with dual model support."""

    def __init__(self, model_size: Optional[str] = None) -> None:
        self._model: Optional[object] = None
        self._rec: Optional[object] = None
        self._model_size = model_size or VOSK_MODEL_SIZE
        self._model_path: Optional[str] = None

    @property
    def name(self) -> str:
        return f"Vosk (offline, {self._model_size})"

    def is_available(self) -> bool:
        if vosk is None:
            return False
        path = self._resolve_model_path()
        return bool(path and Path(path).exists())

    def _resolve_model_path(self) -> str:
        """Resolve model path with auto-download."""
        if self._model_path:
            return self._model_path
        
        path = get_vosk_model_path(self._model_size)
        if path and Path(path).exists():
            self._model_path = path
            return path
        
        # Try explicit path from config
        if VOSK_MODEL_PATH and Path(VOSK_MODEL_PATH).exists():
            self._model_path = VOSK_MODEL_PATH
            return VOSK_MODEL_PATH
        
        return ""

    def _load_model(self) -> bool:
        """Load Vosk model lazily."""
        if self._model is not None:
            return True
        
        path = self._resolve_model_path()
        if not path:
            log.error("Vosk model not found. Set VOSK_MODEL_PATH or enable VOSK_AUTO_DOWNLOAD")
            return False
        
        try:
            import vosk
            log.info("Loading Vosk model from: %s", path)
            self._model = vosk.Model(path)
            self._rec = vosk.KaldiRecognizer(self._model, VOSK_SAMPLE_RATE)
            self._rec.SetWords(True)
            return True
        except Exception as e:
            log.error("Failed to load Vosk model: %s", e)
            return False

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe audio file using Vosk."""
        import wave
        import json
        
        # Check silence threshold
        if self._wav_file_rms(audio_path) <= TRANSCRIPTION_SILENCE_RMS:
            log.info("Audio snippet is below silence threshold; skipping transcription.")
            return ""

        if not self._load_model():
            return ""

        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    log.warning("Vosk requires mono 16-bit PCM WAV")
                    return ""
                
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if self._rec.AcceptWaveform(data):
                        import json
                        result = json.loads(self._rec.Result())
                        if result.get("text"):
                            results.append(result["text"])
                
                # Get final result
                final = json.loads(self._rec.FinalResult())
                if final.get("text"):
                    results.append(final["text"])
                
                return " ".join(results).strip()
        except Exception as e:
            log.error("Vosk transcription error: %s", e)
            return ""

    def _wav_file_rms(self, path: Path) -> float:
        """Compute RMS of a WAV file."""
        import wave
        import numpy as np
        try:
            with wave.open(str(path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
        except (OSError, wave.Error):
            return 0.0
        if not raw:
            return 0.0
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples**2)))