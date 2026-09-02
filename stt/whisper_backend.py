#!/usr/bin/env python3
"""
Faster-Whisper STT backend.

Implements the STTEngine interface using faster-whisper with
automatic CUDA -> CPU fallback at load time and runtime.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import wave

from core.config import (
    TRANSCRIPTION_SILENCE_RMS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from core.logging import log
from stt.interface import STTEngine


@contextmanager
def _without_sslkeylogfile():
    """Temporarily disable inherited SSL key logging that can break sandboxed HTTP clients."""
    prev = os.environ.pop("SSLKEYLOGFILE", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["SSLKEYLOGFILE"] = prev


def _ensure_cuda_dlls():
    """Ensure CUDA DLLs (cublas, cudnn) are findable by CTranslate2 at runtime."""
    import sys
    
    # Pre-load cublas64_12.dll via ctypes to ensure it's loaded in process
    project_root = Path(__file__).resolve().parent.parent
    cublas_path = project_root / "cublas64_12.dll"
    if cublas_path.exists():
        try:
            import ctypes
            ctypes.CDLL(str(cublas_path))
        except Exception:
            pass
    
    # Use os.add_dll_directory (Windows 3.8+) for proper DLL search path
    if hasattr(os, 'add_dll_directory'):
        # Add project root (where cublas64_12.dll is copied)
        project_root = Path(__file__).resolve().parent.parent
        try:
            os.add_dll_directory(str(project_root))
        except Exception:
            pass
        
        # Also add venv root where we copied the DLL
        venv_root = Path(sys.prefix)
        try:
            os.add_dll_directory(str(venv_root))
        except Exception:
            pass
        
        # Add CUDA bin paths
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\x64",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
        ]
        for cuda_path in cuda_paths:
            if os.path.isdir(cuda_path):
                try:
                    os.add_dll_directory(cuda_path)
                except Exception:
                    pass
    
    # Also update PATH for subprocesses
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(project_root) + os.pathsep + os.environ.get("PATH", "")
    
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\x64",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    ]
    for cuda_path in cuda_paths:
        if os.path.isdir(cuda_path) and cuda_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")


class WhisperModelManager:
    """Lazy-loaded singleton for the faster-whisper model."""

    _model = None
    _active_device: str | None = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            last_error: Exception | None = None
            for idx, device in enumerate(cls._candidate_devices()):
                try:
                    log.info("Loading faster-whisper model %s on %s", WHISPER_MODEL_SIZE, device)
                    with _without_sslkeylogfile():
                        from faster_whisper import WhisperModel

                        cls._model = WhisperModel(
                            WHISPER_MODEL_SIZE,
                            device=device,
                            compute_type=WHISPER_COMPUTE_TYPE,
                        )
                    cls._active_device = device
                    if idx > 0:
                        log.warning("Whisper device fallback activated: using %s", device)
                    break
                except Exception as e:
                    last_error = e
                    log.warning("Could not load faster-whisper on %s: %s", device, e)
            if cls._model is None:
                assert last_error is not None
                raise last_error
        return cls._model

    @classmethod
    def reset_model(cls) -> None:
        cls._model = None
        cls._active_device = None

    @staticmethod
    def _candidate_devices() -> list[str]:
        requested = WHISPER_DEVICE or "auto"
        if requested == "auto":
            return ["cuda", "cpu"]
        if requested == "cuda":
            return ["cuda", "cpu"]
        return [requested]


def _wav_file_rms(path: Path) -> float:
    """Compute RMS of a WAV file (int16 PCM). Returns 0.0 on error."""
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


class WhisperSTT(STTEngine):
    """Faster-Whisper STT engine with CUDA->CPU fallback."""

    def __init__(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return f"faster-whisper ({WHISPER_MODEL_SIZE})"

    def is_available(self) -> bool:
        try:
            self._model = WhisperModelManager.get_model()
            return True
        except Exception as e:
            log.error("WhisperSTT unavailable: %s", e)
            return False

    def transcribe(self, audio_path: Path) -> str:
        _ensure_cuda_dlls()
        
        if _wav_file_rms(audio_path) <= TRANSCRIPTION_SILENCE_RMS:
            log.info("Audio snippet is below silence threshold; skipping transcription.")
            return ""

        model = WhisperModelManager.get_model()
        try:
            segments, _info = model.transcribe(str(audio_path), language="en")
            text = " ".join(segment.text for segment in segments)
        except RuntimeError as e:
            # Runtime CUDA failure (e.g., cublas64_12.dll missing during iteration)
            if WhisperModelManager._active_device == "cuda":
                log.warning("Whisper CUDA transcription failed; retrying on CPU: %s", e)
                original_device = WHISPER_DEVICE
                WhisperModelManager.reset_model()
                try:
                    # Force CPU for this retry
                    os.environ["WHISPER_DEVICE"] = "cpu"
                    model = WhisperModelManager.get_model()
                    segments, _info = model.transcribe(str(audio_path), language="en")
                    text = " ".join(segment.text for segment in segments)
                finally:
                    os.environ["WHISPER_DEVICE"] = original_device
            else:
                raise
        return text.strip()