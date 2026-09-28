#!/usr/bin/env python3
"""
Clap detection logic.

Implements a double-clap detector based on RMS energy spikes
relative to an adaptive noise floor.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from core.config import (
    BLOCK_MS,
    COOLDOWN_S,
    MAX_DOUBLE_GAP_S,
    MIN_DOUBLE_GAP_S,
    MIN_RMS,
    NOISE_FLOOR_ALPHA,
    QUIET_GATE_MULT,
    RETRIGGER_RATIO,
    SPIKE_RATIO,
    block_samples,
)
from core.logging import log


class ClapDetector:
    """
    Stateful double-clap detector.

    Feed audio blocks via `process_block()`. When a double clap is detected,
    `on_double_clap` callback is invoked (if set).
    """

    def __init__(
        self,
        on_double_clap: Callable[[float], None] | None = None,
        sample_rate: int = 44100,
        block_ms: int = BLOCK_MS,
    ) -> None:
        self.on_double_clap = on_double_clap
        self.sample_rate = sample_rate
        self.block_ms = block_ms

        # Internal state
        self._audio_buffer: list[np.ndarray] = []
        self._noise_floor = MIN_RMS
        self._last_clap_time = 0.0
        self._cooldown_until = 0.0

        # Max blocks to keep in rolling buffer (~1 second)
        self._max_blocks = int(1000 / block_ms)

    @staticmethod
    def _rms_mono(block: np.ndarray) -> float:
        """Compute RMS of a mono float64 block."""
        if block.ndim > 1:
            block = np.mean(block.astype(np.float64), axis=1)
        else:
            block = block.astype(np.float64)
        if block.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(block**2)))

    def process_block(self, indata: np.ndarray) -> None:
        """
        Process one audio block from sounddevice callback.

        `indata` shape: (frames, channels). Only first channel is used.
        """
        # Extract mono channel as float64
        block = indata[:, 0].astype(np.float64)
        rms = self._rms_mono(block)

        # Update noise floor (slow adaptation during quiet periods)
        if rms < self._noise_floor * QUIET_GATE_MULT:
            self._noise_floor = (
                self._noise_floor * NOISE_FLOOR_ALPHA + rms * (1 - NOISE_FLOOR_ALPHA)
            )

        # Detect clap (spike above noise floor)
        if rms > self._noise_floor * SPIKE_RATIO and rms > MIN_RMS:
            now = time.time()

            # Check for double clap within time window
            gap = now - self._last_clap_time
            if MIN_DOUBLE_GAP_S <= gap <= MAX_DOUBLE_GAP_S:
                if now >= self._cooldown_until:
                    log.info("Double clap detected at %.2f s", now)
                    if self.on_double_clap:
                        self.on_double_clap(now)
                    self._cooldown_until = now + COOLDOWN_S

            self._last_clap_time = now

        # Keep rolling buffer for retrigger logic (if needed later)
        self._audio_buffer.append(block)
        if len(self._audio_buffer) > self._max_blocks:
            self._audio_buffer.pop(0)

    def reset(self) -> None:
        """Reset detector state."""
        self._audio_buffer.clear()
        self._noise_floor = MIN_RMS
        self._last_clap_time = 0.0
        self._cooldown_until = 0.0