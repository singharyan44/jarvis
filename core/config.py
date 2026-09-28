#!/usr/bin/env python3
"""
Central configuration module for JARVIS.

Loads settings from config.json and API keys from .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (for API keys/secrets only)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Load config.json
CONFIG_PATH = BASE_DIR / "config.json"
with open(CONFIG_PATH, "r") as f:
    _config = json.load(f)

# Helper to get nested config with defaults
def _get(path: str, default=None):
    """Get nested config value using dot notation."""
    keys = path.split(".")
    value = _config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value if value is not None else default


# --------------------------------------------------------------------------- #
# Audio / Clap Detection
# --------------------------------------------------------------------------- #
SAMPLE_RATE: int = _get("audio.sample_rate", 44100)
BLOCK_MS: int = _get("audio.block_ms", 40)
CHANNELS: int = _get("audio.channels", 1)

SPIKE_RATIO: float = _get("clap_detection.spike_ratio", 7.0)
COOLDOWN_S: float = _get("clap_detection.cooldown_s", 0.45)
MIN_DOUBLE_GAP_S: float = _get("clap_detection.min_double_gap_s", 0.05)
MAX_DOUBLE_GAP_S: float = _get("clap_detection.max_double_gap_s", 0.35)
RETRIGGER_RATIO: float = _get("clap_detection.retrigger_ratio", 0.55)
NOISE_FLOOR_ALPHA: float = _get("clap_detection.noise_floor_alpha", 0.992)
MIN_RMS: float = _get("clap_detection.min_rms", 0.012)
QUIET_GATE_MULT: float = _get("clap_detection.quiet_gate_mult", 2.2)


def block_samples() -> int:
    """Number of samples per analysis block."""
    n = int(SAMPLE_RATE * BLOCK_MS / 1000)
    return max(n, 1)


# --------------------------------------------------------------------------- #
# Recording / STT
# --------------------------------------------------------------------------- #
RECORD_DURATION_S: float = 3.0

# Voice Activity Detection (VAD) for listen-until-silence recording
VAD_SILENCE_THRESHOLD_RMS: float = _get("vad.silence_threshold_rms", 0.008)
VAD_SILENCE_DURATION_S: float = _get("vad.silence_duration_s", 0.8)
VAD_MIN_RECORDING_S: float = _get("vad.min_recording_s", 1.0)
VAD_MAX_RECORDING_S: float = _get("vad.max_recording_s", 30.0)
VAD_PADDING_S: float = _get("vad.padding_s", 0.3)

# Device for faster-whisper model loading
WHISPER_DEVICE: str = _get("whisper.device", "auto")
WHISPER_COMPUTE_TYPE: str = _get("whisper.compute_type", "default")
WHISPER_MODEL_SIZE: str = _get("whisper.model_size", "small")
TRANSCRIPTION_SILENCE_RMS: float = _get("transcription_silence_rms", 0.0015)


# --------------------------------------------------------------------------- #
# TTS Provider Fallback Order
# --------------------------------------------------------------------------- #
TTS_PREFERENCE_ORDER: list[str] = _get("tts.preference_order", [
    "elevenlabs", "azure", "google", "coqui", "pyttsx3"
])

JARVIS_WELCOME_ENABLED: bool = _get("tts.welcome_enabled", True)
JARVIS_WELCOME_PHRASE: str = _get("tts.welcome_phrase", "Welcome home sir. ")
JARVIS_AFTER_SONG_DELAY_S: float = _get("tts.after_song_delay_s", 1.0)
JARVIS_WELCOME_CACHE_ENABLED: bool = _get("tts.welcome_cache_enabled", True)


# --------------------------------------------------------------------------- #
# Spotify / Media
# --------------------------------------------------------------------------- #
SONG_URI: str = _get("media.song_uri", "https://open.spotify.com/track/39shmbIHICJ2Wxnk1fPSdz?si=2900c75c2e2d4b82")


# --------------------------------------------------------------------------- #
# Cursor / Editor
# --------------------------------------------------------------------------- #
FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP: bool = _get("cursor.focus_existing", False)
OPEN_NEW_CURSOR_ON_DOUBLE_CLAP: bool = _get("cursor.open_new", False)
CURSOR_OPEN_FULLSCREEN: bool = _get("cursor.fullscreen", False)


# --------------------------------------------------------------------------- #
# Google Chrome
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Welcome Phrase / ElevenLabs Caching
# --------------------------------------------------------------------------- #
JARVIS_WELCOME_ENABLED: bool = True
JARVIS_WELCOME_PHRASE: str = "Welcome home sir. "
JARVIS_AFTER_SONG_DELAY_S: float = 1.0
JARVIS_WELCOME_CACHE_ENABLED: bool = True


# --------------------------------------------------------------------------- #
# ElevenLabs Runtime Config Helpers
# --------------------------------------------------------------------------- #
def _elevenlabs_pcm_sample_rate(output_format: str) -> int:
    """Extract sample rate from output_format like 'pcm_24000'."""
    override = os.environ.get("ELEVENLABS_PCM_SAMPLE_RATE", "").strip()
    if override.isdigit():
        return int(override)
    if output_format.startswith("pcm_"):
        try:
            return int(output_format.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            pass
    return 24000


def elevenlabs_env_config() -> tuple[str, str, str, int]:
    """Returns (voice_id, model_id, output_format, pcm_sample_rate) from env."""
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
    fmt = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000").strip()
    rate = _elevenlabs_pcm_sample_rate(fmt)
    return voice, model, fmt, rate


def jarvis_welcome_cache_dir() -> Path:
    """Directory where cached welcome WAV files are stored."""
    override = os.environ.get("JARVIS_WELCOME_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return BASE_DIR / ".cache" / "jarvis_welcome"


def jarvis_welcome_cache_path(
    text: str, voice_id: str, model_id: str, output_format: str
) -> Path:
    """Deterministic cache path for a given text+voice+model+format combination."""
    import hashlib
    key = f"{text}|{voice_id}|{model_id}|{output_format}".encode()
    digest = hashlib.sha256(key).hexdigest()[:24]
    return jarvis_welcome_cache_dir() / f"{digest}.wav"


# --------------------------------------------------------------------------- #
# AI Intent Layer (Phase 5) - OpenRouter Configuration
# --------------------------------------------------------------------------- #
# OpenRouter API key for AI intent classification (from .env)
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "").strip()
# Model to use for intent classification (set in config.json)
OPENROUTER_MODEL: str = _get("openrouter.model", "")
# Base URL for OpenRouter API
OPENROUTER_BASE_URL: str = _get("openrouter.base_url", "https://openrouter.ai/api/v1")
# Timeout for API calls
OPENROUTER_TIMEOUT: float = _get("openrouter.timeout", 10.0)
# Temperature for intent classification (low for consistency)
OPENROUTER_TEMPERATURE: float = _get("openrouter.temperature", 0.1)
# Max tokens for response
OPENROUTER_MAX_TOKENS: int = _get("openrouter.max_tokens", 500)


# --------------------------------------------------------------------------- #
# Vosk STT (Offline) Configuration
# --------------------------------------------------------------------------- #
VOSK_MODEL_SIZE: str = _get("vosk.model_size", "small")
VOSK_MODEL_PATH: str = os.environ.get("VOSK_MODEL_PATH", "").strip()
VOSK_SAMPLE_RATE: int = _get("vosk.sample_rate", 16000)
VOSK_AUTO_DOWNLOAD: bool = _get("vosk.auto_download", True)


# --------------------------------------------------------------------------- #
# Local LLM Configuration (Offline Mode)
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL: str = _get("local_llm.ollama_base_url", "http://localhost:11434")
OLLAMA_MODEL: str = _get("local_llm.ollama_model", "llama3.1:8b")
LM_STUDIO_BASE_URL: str = _get("local_llm.lm_studio_base_url", "http://localhost:1234")
LM_STUDIO_MODEL: str = _get("local_llm.lm_studio_model", "")


# --------------------------------------------------------------------------- #
# Offline Mode Configuration
# --------------------------------------------------------------------------- #
OFFLINE_MODE: str = _get("offline_mode", "auto")


# --------------------------------------------------------------------------- #
# Honcho Configuration
# --------------------------------------------------------------------------- #
HONCHO_API_URL: str = _get("honcho.api_url", "http://localhost:8000")
HONCHO_WORKSPACE: str = _get("honcho.workspace", "jarvis")
HONCHO_ENABLED: bool = _get("honcho.enabled", True)


# --------------------------------------------------------------------------- #
# Wake Word Configuration
# --------------------------------------------------------------------------- #
WAKE_WORD_ENABLED: bool = _get("wake_word.enabled", True)
WAKE_WORD_PHRASES: list[str] = _get("wake_word.phrases", ["hey jarvis", "jarvis"])
WAKE_WORD_SENSITIVITY: float = _get("wake_word.sensitivity", 0.5)
WAKE_WORD_COOLDOWN: float = _get("wake_word.cooldown", 2.0)


def is_offline_mode() -> bool:
    """Determine if we should run in offline mode."""
    if OFFLINE_MODE == "true":
        return True
    if OFFLINE_MODE == "false":
        return False
    # Auto-detect
    from core.network import is_online
    return not is_online()


def get_vosk_model_path(model_size: str | None = None) -> str:
    """Get Vosk model path, auto-downloading if needed."""
    size = model_size or VOSK_MODEL_SIZE
    if VOSK_MODEL_PATH:
        return VOSK_MODEL_PATH
    if VOSK_AUTO_DOWNLOAD:
        from core.network import get_vosk_model_path as _get_vosk_model_path
        return _get_vosk_model_path(size) or ""
    return ""


# --------------------------------------------------------------------------- #
# Helpers for API keys from .env
# --------------------------------------------------------------------------- #
def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ElevenLabs
ELEVENLABS_API_KEY: str = _get_env("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID: str = _get_env("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID: str = _get_env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT: str = _get_env("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000")


def _elevenlabs_pcm_sample_rate(output_format: str) -> int:
    override = os.environ.get("ELEVENLABS_PCM_SAMPLE_RATE", "").strip()
    if override.isdigit():
        return int(override)
    if output_format.startswith("pcm_"):
        try:
            return int(output_format.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            pass
    return 24000


def elevenlabs_env_config() -> tuple[str, str, str, int]:
    voice = ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID and ELEVENLABS_VOICE_ID or ""
    voice = _get_env("ELEVENLABS_VOICE_ID")
    model = _get_env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    fmt = _get_env("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000")
    rate = _elevenlabs_pcm_sample_rate(fmt)
    return voice, model, fmt, rate


def jarvis_welcome_cache_dir() -> Path:
    override = os.environ.get("JARVIS_WELCOME_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return BASE_DIR / ".cache" / "jarvis_welcome"


def jarvis_welcome_cache_path(
    text: str, voice_id: str, model_id: str, output_format: str
) -> Path:
    import hashlib
    key = f"{text}|{voice_id}|{model_id}|{output_format}".encode()
    digest = hashlib.sha256(key).hexdigest()[:24]
    return jarvis_welcome_cache_dir() / f"{digest}.wav"