#!/usr/bin/env python3
"""
Audio capture and playback utilities.

Provides functions to record audio from the default microphone
and play raw PCM audio through the default output device.

Supports both fixed-duration recording and listen-until-silence (VAD) recording.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from core.config import (
    CHANNELS,
    SAMPLE_RATE,
    VAD_MAX_RECORDING_S,
    VAD_MIN_RECORDING_S,
    VAD_PADDING_S,
    VAD_SILENCE_DURATION_S,
    VAD_SILENCE_THRESHOLD_RMS,
)


def _rms_f32(block: np.ndarray) -> float:
    """Compute RMS of a float32 mono block."""
    if block.ndim > 1:
        block = np.mean(block.astype(np.float64), axis=1)
    else:
        block = block.astype(np.float64)
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))


def record_audio(duration: float) -> Path:
    """
    Record `duration` seconds from the default microphone (fixed duration).

    Returns a temporary WAV file path (int16 PCM, mono, SAMPLE_RATE).
    The caller is responsible for deleting the file when done.
    """
    fs = SAMPLE_RATE
    samples = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype="float32")
    sd.wait()

    # Convert float32 [-1, 1] to int16 PCM for Whisper
    pcm_i16 = (samples * 32767).astype("int16")

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    tmp = Path(tmp_path)

    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm_i16.tobytes())

    tmp.replace(tmp_path)
    return Path(tmp_path)


def record_until_silence(
    silence_threshold_rms: float | None = None,
    silence_duration_s: float | None = None,
    min_recording_s: float | None = None,
    max_recording_s: float | None = None,
    padding_s: float | None = None,
) -> Path:
    """
    Record audio until silence is detected (Voice Activity Detection).

    Args:
        silence_threshold_rms: RMS level below which audio is considered silent.
        silence_duration_s: Seconds of continuous silence to stop recording.
        min_recording_s: Minimum recording duration before silence detection activates.
        max_recording_s: Maximum recording duration (safety timeout).
        padding_s: Seconds of audio to keep before/after detected speech.

    Returns:
        Path to temporary WAV file (int16 PCM, mono, SAMPLE_RATE).
    """
    # Use config defaults if not provided
    silence_threshold_rms = silence_threshold_rms or VAD_SILENCE_THRESHOLD_RMS
    silence_duration_s = silence_duration_s or VAD_SILENCE_DURATION_S
    min_recording_s = min_recording_s or VAD_MIN_RECORDING_S
    max_recording_s = max_recording_s or VAD_MAX_RECORDING_S
    padding_s = padding_s or VAD_PADDING_S

    fs = SAMPLE_RATE
    block_size = int(fs * 0.1)  # 100ms blocks for VAD
    silence_blocks_needed = int(silence_duration_s / 0.1)
    min_blocks = int(min_recording_s / 0.1)
    max_blocks = int(max_recording_s / 0.1)
    padding_blocks = int(padding_s / 0.1)

    audio_chunks = []
    silence_blocks = 0
    speech_detected = False

    def callback(indata, frames, time_info, status):
        nonlocal silence_blocks, speech_detected
        if status:
            print(f"Audio callback status: {status}")

        # indata is (frames, channels) - take first channel
        block = indata[:, 0].copy()
        rms = _rms_f32(block)
        audio_chunks.append(block)

        if rms > silence_threshold_rms:
            speech_detected = True
            silence_blocks = 0
        elif speech_detected:
            silence_blocks += 1

    # Start recording stream
    with sd.InputStream(
        samplerate=fs,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        while True:
            sd.sleep(100)  # Check every 100ms

            # Stop conditions
            if len(audio_chunks) >= max_blocks:
                break
            if speech_detected and silence_blocks >= silence_blocks_needed and len(audio_chunks) >= min_blocks:
                break

    # Concatenate all chunks
    if not audio_chunks:
        # Return empty recording (min duration of silence)
        return record_audio(min_recording_s)

    full_audio = np.concatenate(audio_chunks)

    # Apply padding: trim silence from end but keep padding_s
    if speech_detected and padding_blocks > 0:
        # Keep padding_blocks worth of audio after last speech
        # (already captured in the loop since we wait for silence_blocks)
        pass

    # Convert to int16 PCM
    pcm_i16 = (full_audio * 32767).astype("int16")

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    tmp = Path(tmp_path)

    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm_i16.tobytes())

    tmp.replace(tmp_path)
    return Path(tmp_path)


def play_pcm_f32(pcm_f32: np.ndarray, sample_rate: int) -> None:
    """
    Play float32 PCM audio (mono, range [-1, 1]) through default output.

    Blocks until playback finishes.
    """
    sd.play(pcm_f32, sample_rate)
    sd.wait()


def play_pcm_i16(pcm_i16: np.ndarray, sample_rate: int) -> None:
    """
    Play int16 PCM audio (mono) through default output.

    Blocks until playback finishes.
    """
    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
    sd.play(pcm_f32, sample_rate)
    sd.wait()