#!/usr/bin/env python3
"""
Speech cleanup layer (Phase 2).

Converts raw STT output into cleaner text:
- Filler word removal (uh, um, like, you know)
- Punctuation insertion
- Capitalization
- Common STT error corrections

This is a placeholder implementation. Phase 2 will enhance this with
more sophisticated NLP-based cleanup.
"""

from __future__ import annotations

import re


# Common filler words to remove
_FILLER_WORDS = {
    "uh", "um", "er", "ah", "like", "you know", "i mean", "sort of",
    "kind of", "basically", "actually", "literally", "so", "well",
    "right", "okay", "ok", "yeah", "yes", "no", "hmm", "hm",
}

# Common STT corrections
_STT_CORRECTIONS = {
    "jarvis": "Jarvis",
    "cursor": "Cursor",
    "chrome": "Chrome",
    "spotify": "Spotify",
    "binance": "Binance",
    "btc": "BTC",
    "claude": "Claude",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "api": "API",
    "url": "URL",
    "http": "HTTP",
    "https": "HTTPS",
    "ai": "AI",
    "ml": "ML",
    "gpu": "GPU",
    "cpu": "CPU",
    "ram": "RAM",
    "ssd": "SSD",
    "os": "OS",
    "ui": "UI",
    "ux": "UX",
}


def clean_transcript(raw_text: str) -> str:
    """
    Clean up raw STT transcript.

    Args:
        raw_text: Raw text from STT engine.

    Returns:
        Cleaned text with filler words removed, proper punctuation,
        capitalization, and common corrections applied.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # 1. Remove filler words (case-insensitive, word boundaries)
    for filler in _FILLER_WORDS:
        # Match filler words with optional punctuation around them
        pattern = rf"\b{re.escape(filler)}\b[,.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 2. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # 4. Ensure sentence ends with punctuation
    if text and text[-1] not in ".!?":
        text += "."

    # 5. Apply common STT corrections
    for wrong, correct in _STT_CORRECTIONS.items():
        # Case-insensitive replacement preserving original case pattern
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)

    # 6. Fix spacing around punctuation
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)

    return text


def clean_for_intent(raw_text: str) -> str:
    """
    Lightweight cleanup for intent classification.

    More aggressive than clean_transcript - removes filler completely,
    normalizes whitespace, lowercases for matching.
    """
    if not raw_text:
        return ""

    text = raw_text.lower().strip()

    # Remove filler words entirely
    for filler in _FILLER_WORDS:
        text = re.sub(rf"\b{re.escape(filler)}\b", "", text, flags=re.IGNORECASE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Confirmation keywords for local yes/no classification
_CONFIRM_YES = {
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm",
    "proceed", "go ahead", "do it", "please", "affirmative",
    "correct", "right", "exactly", "do that"
}

_CONFIRM_NO = {
    "no", "nope", "cancel", "stop", "abort", "dont", "don't",
    "never mind", "nevermind", "wait", "hold on", "negative",
    "wrong", "incorrect", "not"
}


def classify_confirmation(text: str) -> str | None:
    """
    Classify a short user response as confirmation (yes/no).
    
    Returns:
        "yes" - user confirmed
        "no" - user denied/cancelled
        None - unclear/ambiguous
    """
    if not text:
        return None
    
    cleaned = clean_for_intent(text)
    if not cleaned:
        return None
    
    # Check for exact matches first (highest confidence)
    for word in _CONFIRM_YES:
        if word == cleaned:
            return "yes"
    for word in _CONFIRM_NO:
        if word == cleaned:
            return "no"
    
    # Check for contains (lower confidence, but catches "yes please", "no thanks")
    for word in _CONFIRM_YES:
        if word in cleaned:
            return "yes"
    for word in _CONFIRM_NO:
        if word in cleaned:
            return "no"
    
    return None