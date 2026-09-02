#!/usr/bin/env python3
"""
Voice Wake Word Detector — uses Vosk for continuous listening.

Listens for wake phrases and triggers callback when detected.
Uses small Vosk model (~40MB) for fast, lightweight detection.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd

from core.config import (
    SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD_RMS,
    VAD_SILENCE_DURATION_S,
    WAKE_WORD_COOLDOWN,
    WAKE_WORD_ENABLED,
    WAKE_WORD_PHRASES,
    WAKE_WORD_SENSITIVITY,
    VOSK_MODEL_SIZE,
    get_vosk_model_path,
)
from core.logging import log


class VoiceWakeDetector:
    """
    Continuous voice wake word detector using Vosk.
    
    Runs a background audio stream and detects wake phrases.
    Uses small Vosk model (~40MB) for fast, lightweight detection.
    """
    
    def __init__(
        self,
        on_wake: Callable[[float, str], None],
        phrases: Optional[list[str]] = None,
        sensitivity: float = 0.5,
        cooldown: float = 2.0,
    ) -> None:
        self.on_wake = on_wake
        self.phrases = phrases or WAKE_WORD_PHRASES
        self.sensitivity = sensitivity
        self.cooldown = cooldown
        
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._last_wake_time = 0.0
        self._model = None
        self._rec = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        
        # Check if Vosk is available
        self._available = self._check_vosk()
        if not self._available:
            log.warning("Voice wake word not available (Vosk not installed or model missing)")
    
    def _check_vosk(self) -> bool:
        """Check if Vosk is available and model exists/auto-downloads."""
        try:
            import vosk
            model_path = get_vosk_model_path(VOSK_MODEL_SIZE)
            if model_path and Path(model_path).exists():
                return True
            return False
        except ImportError:
            return False
    
    def start(self) -> bool:
        """Start the wake word detector."""
        if not self._available:
            log.warning("Cannot start wake detector: Vosk not available")
            return False
        
        if self._running:
            return True
        
        # Load Vosk model
        try:
            import vosk
            model_path = get_vosk_model_path(VOSK_MODEL_SIZE)
            if not model_path or not Path(model_path).exists():
                log.error("Vosk model not found at %s", model_path)
                return False
            
            log.info("Loading Vosk model for wake word: %s", model_path)
            self._model = vosk.Model(model_path)
            self._rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
            self._rec.SetWords(True)
        except Exception as e:
            log.error("Failed to load Vosk model: %s", e)
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        
        # Start audio stream
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=4000,
                callback=self._audio_callback,
            )
            self._stream.start()
            log.info("Voice wake word detector started (phrases: %s)", self.phrases)
            return True
        except Exception as e:
            log.error("Failed to start audio stream: %s", e)
            self._running = False
            return False
    
    def stop(self) -> None:
        """Stop the wake word detector."""
        self._running = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        log.info("Voice wake word detector stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            log.debug("Audio status: %s", status)
        self._audio_queue.put(bytes(indata))
    
    def _processing_loop(self) -> None:
        """Background processing of audio for wake word detection."""
        import json
        
        while self._running:
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if not self._running:
                break
            
            if self._rec.AcceptWaveform(data):
                result = json.loads(self._rec.Result())
                text = result.get("text", "").lower().strip()
                
                if text:
                    log.debug("Wake detector heard: %s", text)
                    confidence = self._get_phrase_confidence(result)
                    if confidence >= self.sensitivity and self._check_wake_word(text):
                        now = time.time()
                        if now - self._last_wake_time >= self.cooldown:
                            log.info("Wake word detected: %s (confidence: %.2f)", text, confidence)
                            self._last_wake_time = now
                            # Trigger callback in a separate thread
                            threading.Thread(
                                target=self.on_wake,
                                args=(now, text),
                                daemon=True
                            ).start()
                    else:
                        log.debug("Wake phrase matched but confidence %.2f < threshold %.2f", confidence, self.sensitivity)
    
    def _get_phrase_confidence(self, result: dict) -> float:
        """
        Extract minimum confidence for any wake phrase from Vosk word-level results.
        
        Vosk returns 'result' array with {word, conf, start, end} when SetWords(True).
        We find the wake phrase words and return their minimum confidence.
        """
        words = result.get("result", [])
        if not words:
            return 0.0
        
        full_text = result.get("text", "").lower()
        best_confidence = 0.0
        
        for phrase in self.phrases:
            phrase_words = phrase.lower().split()
            if not phrase_words:
                continue
            
            # Find contiguous match of phrase words in recognized words
            for i in range(len(words) - len(phrase_words) + 1):
                match = True
                phrase_confidences = []
                
                for j, pw in enumerate(phrase_words):
                    if i + j >= len(words):
                        match = False
                        break
                    # Compare normalized (lowercase, no punctuation)
                    recognized_word = words[i + j]["word"].lower().strip(".,!?")
                    if recognized_word != pw:
                        match = False
                        break
                    phrase_confidences.append(words[i + j]["conf"])
                
                if match and phrase_confidences:
                    min_conf = min(phrase_confidences)
                    best_confidence = max(best_confidence, min_conf)
        
        return best_confidence
    
    def _check_wake_word(self, text: str) -> bool:
        """Check if any wake phrase is in the recognized text (substring match)."""
        text = text.lower().strip()
        for phrase in self.phrases:
            if phrase in text:
                return True
        return False


# Global instance
_voice_wake_detector: Optional[VoiceWakeDetector] = None


def get_voice_wake_detector(on_wake: Callable[[float, str], None]) -> Optional[VoiceWakeDetector]:
    """Get or create the global voice wake detector."""
    global _voice_wake_detector
    if not WAKE_WORD_ENABLED:
        return None
    if _voice_wake_detector is None:
        _voice_wake_detector = VoiceWakeDetector(on_wake=on_wake)
    return _voice_wake_detector