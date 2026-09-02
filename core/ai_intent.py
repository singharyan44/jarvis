#!/usr/bin/env python3
"""
AI Intent Client — OpenRouter-based intent classification for JARVIS.

Takes cleaned transcript text and returns structured Intent for command routing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from commands.registry import Intent
from core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT,
)
from core.logging import log


@dataclass
class IntentClassification:
    """Result of intent classification."""
    intent: Intent | None
    raw_response: str
    latency: float
    success: bool
    error: str = ""


# System prompt for intent classification
INTENT_SYSTEM_PROMPT = """You are JARVIS, a voice-controlled personal assistant. Your task is to classify the user's speech transcript into a structured intent for command execution.

Available commands (tools):
- play_song: Play configured song/media. Actions: play (default). Parameters: uri (string, optional)
- execute_shell: Execute a shell command. Actions: run (default), execute. Parameters: command (string, required)

Respond ONLY with valid JSON in this exact format:
{
  "intent": "command",
  "tool": "tool_name",
  "action": "action_name",
  "parameters": {},
  "confidence": 0.95,
  "response": "Natural language response to speak to user"
}

If the transcript doesn't match any command, return:
{
  "intent": "none",
  "tool": "none",
  "action": "none",
  "parameters": {},
  "confidence": 0.0,
  "response": "I didn't understand that command."
}

Rules:
- intent_type is always "command" for recognized commands, "none" for unrecognized
- confidence reflects how certain you are (0.0-1.0)
- response is what JARVIS should say to the user
- parameters must match the tool's expected parameters"""


def _build_user_prompt(transcript: str) -> str:
    """Build the user prompt with the transcript."""
    return f'User said: "{transcript}"\n\nClassify this into a structured intent.'


def _parse_ai_response(response_text: str) -> Intent | None:
    """Parse the AI response into an Intent object."""
    try:
        data = json.loads(response_text.strip())
        
        # Validate required fields
        required = ["intent", "tool", "action", "parameters", "confidence", "response"]
        if not all(k in data for k in required):
            log.warning("AI response missing required fields: %s", response_text)
            return None
        
        if data["intent"] == "none" or data["tool"] == "none":
            return None
        
        return Intent(
            intent_type=data["intent"],
            tool=data["tool"],
            action=data["action"],
            parameters=data.get("parameters", {}),
            confidence=float(data.get("confidence", 0.0)),
            raw_transcript="",  # Will be filled by caller
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("Failed to parse AI response: %s - %s", e, response_text)
        return None


class AIIntentClient:
    """
    Client for classifying transcripts into structured intents using OpenRouter.
    
    Usage:
        client = AIIntentClient()
        intent = await client.classify("play some music")
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Use explicit None check to allow empty string override
        self.api_key = api_key if api_key is not None else OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.timeout = timeout or OPENROUTER_TIMEOUT
        
        if not self.api_key:
            log.warning("OPENROUTER_API_KEY not configured; AI intent classification will not work")
    
    def is_available(self) -> bool:
        """Check if the client is configured and ready."""
        return bool(self.api_key)
    
    async def classify(self, transcript: str) -> IntentClassification:
        """
        Classify a transcript into a structured intent.
        
        Args:
            transcript: Cleaned transcript text from STT + cleanup layer
            
        Returns:
            IntentClassification with parsed Intent or error info
        """
        if not self.is_available():
            return IntentClassification(
                intent=None,
                raw_response="",
                latency=0.0,
                success=False,
                error="OpenRouter API key not configured"
            )
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://jarvis.local",
                        "X-Title": "JARVIS Assistant",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                            {"role": "user", "content": _build_user_prompt(transcript)},
                        ],
                        "temperature": OPENROUTER_TEMPERATURE,
                        "max_tokens": OPENROUTER_MAX_TOKENS,
                    }
                )
                
                latency = time.time() - start_time
                
                if response.status_code != 200:
                    error_text = response.text
                    log.error("OpenRouter API error %s: %s", response.status_code, error_text)
                    return IntentClassification(
                        intent=None,
                        raw_response=error_text,
                        latency=latency,
                        success=False,
                        error=f"API error {response.status_code}"
                    )
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Log AI call for analytics
                log.info("AI intent classification: %s -> %s", transcript[:50], content[:100])
                
                intent = _parse_ai_response(content)
                
                if intent:
                    intent.raw_transcript = transcript
                
                return IntentClassification(
                    intent=intent,
                    raw_response=content,
                    latency=latency,
                    success=intent is not None,
                    error="" if intent else "Failed to parse intent from response"
                )
                
        except httpx.TimeoutException:
            latency = time.time() - start_time
            log.error("OpenRouter API timeout after %.1fs", latency)
            return IntentClassification(
                intent=None,
                raw_response="",
                latency=latency,
                success=False,
                error="API timeout"
            )
        except Exception as e:
            latency = time.time() - start_time
            log.error("OpenRouter API error: %s", e)
            return IntentClassification(
                intent=None,
                raw_response="",
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def classify_sync(self, transcript: str) -> IntentClassification:
        """Synchronous version for non-async contexts."""
        import asyncio
        try:
            return asyncio.run(self.classify(transcript))
        except RuntimeError:
            # If already in an event loop, create a new task
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.classify(transcript))
            finally:
                loop.close()


# Convenience function for simple usage
async def classify_intent(transcript: str) -> Intent | None:
    """
    Classify transcript and return Intent if successful.
    
    Returns None if classification fails or no command matched.
    """
    client = AIIntentClient()
    result = await client.classify(transcript)
    return result.intent