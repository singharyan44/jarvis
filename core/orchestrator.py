#!/usr/bin/env python3
"""
Core orchestrator — ties together audio, STT, TTS, and commands.

This is the main application logic that coordinates all subsystems.
"""

from __future__ import annotations

import asyncio
import threading
import time

from core.config import (
    JARVIS_AFTER_SONG_DELAY_S,
    JARVIS_WELCOME_ENABLED,
    JARVIS_WELCOME_PHRASE,
)
from core.logging import log
from core.safety import SafetyAnalyzer, get_safety_analyzer
from audio.wake_word import VoiceWakeDetector, get_voice_wake_detector
from cleanup.cleaner import clean_transcript, classify_confirmation
from commands.registry import CommandRegistry, Intent
from commands.system_commands import PlaySongCommand, ExecuteShellCommand
from core.ai_intent import AIIntentClient, IntentClassification
from core.validator import ActionValidator, ValidationResult, ConfirmationManager
from logs import get_session_logger, close_session_logger
from stt.whisper_backend import WhisperSTT
from tts.manager import TTSManager
from memory import get_memory_manager, get_context_manager


class JarvisOrchestrator:
    """
    Main JARVIS application orchestrator.

    Coordinates:
    - Wake word detection (audio input)
    - Speech-to-text (Whisper)
    - Command execution (registry) with AI intent fallback
    - Validation layer (Phase 6)
    - Text-to-speech (fallback chain)
    - Session logging (JSONL)
    """

    def __init__(self) -> None:
        # Initialize subsystems
        self._stt = WhisperSTT()
        self._tts = TTSManager()
        self._commands = CommandRegistry()
        self._session_logger = get_session_logger()
        self._ai_intent = AIIntentClient()
        self._validator = ActionValidator(self._commands)
        self._confirmation = ConfirmationManager(self._tts)

        # Safety Analyzer (Phase 11 - Terminal Safety)
        self._safety_analyzer = get_safety_analyzer()

        # Memory & Context (Phase 9-10)
        self._memory = get_memory_manager()
        self._context = get_context_manager()

        # Session management
        self._session_id: str | None = None

        # Wake word detector (replaces clap detector)
        self._wake_detector = get_voice_wake_detector(on_wake=self._on_wake_word)
        
        # State management to prevent re-triggering
        self._is_listening = False
        self._is_processing = False

        # Pending confirmation state (for dangerous actions)
        self._pending_confirmation: dict | None = None  # {"intent": Intent, "ai_response": str, "start_time": float, "transcript": str}

        # Register built-in commands
        self._register_commands()

        # State
        self._running = False

    def _register_commands(self) -> None:
        """Register all built-in commands."""
        self._commands.register(PlaySongCommand())
        self._commands.register(ExecuteShellCommand())

    def _on_wake_word(self, timestamp: float, phrase: str) -> None:
        """Callback when wake word is detected."""
        # Prevent re-triggering while listening or processing
        if self._is_listening or self._is_processing:
            log.debug("Wake word ignored - already listening/processing")
            return
        
        log.info("Wake word detected: %s", phrase)
        threading.Thread(target=self._run_wake_actions, daemon=True).start()

    def _run_wake_actions(self) -> None:
        """Handle wake word: record, transcribe, clean up, execute command."""
        # Set state to prevent re-triggering
        self._is_listening = True
        self._is_processing = True
        
        audio_path = None
        try:
            log.info("Wake word triggered – recording audio until silence.")

            # Lazy import to avoid circular dependency
            from audio.capture import record_until_silence
            audio_path = record_until_silence()
            
            # Recording done, now processing
            self._is_listening = False
            
            start_time = time.time()
            raw_transcript = self._stt.transcribe(audio_path)
            stt_latency = time.time() - start_time

            log.info("Transcribed (raw): %s", raw_transcript)

            # Log STT performance
            self._session_logger.log_performance(
                operation="stt_transcribe",
                latency=stt_latency,
                success=bool(raw_transcript),
                metadata={"audio_path": str(audio_path), "raw_length": len(raw_transcript)}
            )

            # Clean up transcript (filler removal, punctuation, corrections)
            transcript = clean_transcript(raw_transcript)
            if transcript != raw_transcript:
                log.info("Transcribed (cleaned): %s", transcript)

            # Handle pending confirmation first
            if self._pending_confirmation:
                self._handle_confirmation_response(transcript, stt_latency)
                return

            if transcript:
                # First try keyword-based intent matching (Phase 3)
                cmd_start = time.time()
                cmd, intent = self._commands.find_matching_with_intent(transcript)
                
                if cmd and intent:
                    # Keyword match succeeded - validate before execution
                    log.info("Keyword match: tool=%s, action=%s, params=%s", intent.tool, intent.action, intent.parameters)
                    validation_result = self._validator.validate(intent)
                    
                    if validation_result.valid:
                        self._execute_validated_intent(transcript, validation_result, cmd_start)
                    else:
                        log.warning("Validation failed for keyword intent: %s", validation_result.error)
                        self._session_logger.log_command(
                            transcript=transcript,
                            tool=intent.tool,
                            action=intent.action,
                            latency=time.time() - cmd_start,
                            success=False,
                            message=f"Validation failed: {validation_result.error}",
                        )
                else:
                    # Keyword match failed - try AI intent classification (Phase 5)
                    log.info("No keyword match; attempting AI intent classification...")
                    cmd_latency = time.time() - cmd_start
                    
                    ai_result = self._classify_with_ai(transcript)
                    if ai_result and ai_result.intent:
                        # AI classification succeeded - validate before execution
                        ai_intent = ai_result.intent
                        ai_intent.raw_transcript = transcript
                        log.info("AI intent: tool=%s, action=%s, params=%s, confidence=%.2f", 
                                ai_intent.tool, ai_intent.action, ai_intent.parameters, ai_intent.confidence)
                        
                        validation_result = self._validator.validate(ai_intent)
                        
                        if validation_result.valid:
                            self._execute_validated_intent(transcript, validation_result, time.time())
                        else:
                            log.warning("Validation failed for AI intent: %s", validation_result.error)
                            self._session_logger.log_command(
                                transcript=transcript,
                                tool=ai_intent.tool,
                                action=ai_intent.action,
                                latency=time.time() - cmd_start,
                                success=False,
                                message=f"Validation failed: {validation_result.error}",
                            )
                    else:
                        # Both keyword and AI failed
                        total_latency = time.time() - cmd_start
                        log.info("AI classification failed or no command matched: %s", ai_result.error if ai_result else "No result")
                        self._session_logger.log_command(
                            transcript=transcript,
                            tool="unknown",
                            action="none",
                            latency=total_latency,
                            success=False,
                            message="No matching command found (keyword + AI)",
                        )
            else:
                log.warning("Empty transcript – no command recognized.")
                self._session_logger.log_command(
                    transcript=raw_transcript,
                    tool="stt",
                    action="empty_transcript",
                    latency=stt_latency,
                    success=False,
                    message="Empty transcript after STT",
                )
        except Exception as e:
            log.error("Error during transcription or command processing: %s", e)
            self._session_logger.log_error(
                error_message=str(e),
                error_type=type(e).__name__,
                context={"stage": "wake_actions"}
            )
        finally:
            # Clean up temporary audio file
            if audio_path:
                try:
                    audio_path.unlink()
                except Exception:
                    pass
            # Reset state flags
            self._is_processing = False
            self._is_listening = False

    def _execute_validated_intent(self, transcript: str, validation_result, start_time: float) -> None:
        """Execute a validated intent, handling confirmation if required."""
        intent = validation_result.intent
        
        # Check if confirmation required
        if validation_result.requires_confirmation:
            log.info("Confirmation required: %s", validation_result.confirmation_message)
            
            # Get the AI's response (confirmation question) from the last AI classification
            # We need to pass it through - for now use the validation message
            ai_response = validation_result.confirmation_message
            
            # Store pending confirmation
            self._pending_confirmation = {
                "intent": intent,
                "ai_response": ai_response,
                "start_time": start_time,
                "transcript": transcript,
            }
            
            # Speak the confirmation question via TTS
            log.info("Asking for confirmation: %s", ai_response)
            self._tts.speak(ai_response)
            
            # Don't reset _is_processing yet - we're waiting for confirmation
            self._is_processing = True
            self._is_listening = False
            return
        
        # SAFETY CHECK: Run safety analyzer for terminal commands
        if intent.tool == "execute_shell":
            from core.safety import ToolRequest
            safety_request = ToolRequest(
                intent=transcript,
                tool=intent.tool,
                shell="bash",
                command=intent.parameters.get("command", ""),
                context={
                    "cwd": "/workspace",
                    "sandbox": "isolated_docker",
                    "network": False,
                    "privileges": "non_root",
                    "scope": "workspace_only",
                },
            )
            
            safety_result = self._safety_analyzer.analyze(safety_request)
            
            log.info("Safety analysis: %s (confidence: %.2f, risk: %.2f) - %s",
                     safety_result.decision.value, safety_result.confidence, 
                     safety_result.risk_score, safety_result.explanation)
            
            if safety_result.decision.value == "DENY":
                log.warning("Command blocked by safety analyzer: %s", safety_result.explanation)
                self._session_logger.log_command(
                    transcript=transcript,
                    tool=intent.tool,
                    action=intent.action,
                    latency=time.time() - start_time,
                    success=False,
                    message=f"Blocked by safety: {safety_result.explanation}",
                    parameters=intent.parameters,
                )
                self._tts.speak(f"Command blocked: {safety_result.explanation}")
                self._is_processing = False
                self._is_listening = False
                return
            elif safety_result.decision.value == "CONFIRM":
                # Store for confirmation flow
                self._pending_confirmation = {
                    "intent": intent,
                    "ai_response": safety_result.explanation,
                    "start_time": start_time,
                    "transcript": transcript,
                }
                log.info("Asking for confirmation: %s", safety_result.explanation)
                self._tts.speak(safety_result.explanation)
                self._is_processing = True
                self._is_listening = False
                return
            # ALLOW falls through to execution
        
        # Execute the command
        cmd = self._commands.get(intent.tool)
        if cmd:
            result = cmd.execute(transcript, **intent.parameters)
            exec_latency = time.time() - start_time
            
            self._session_logger.log_command(
                transcript=transcript,
                tool=intent.tool,
                action=intent.action,
                latency=exec_latency,
                success=result.success,
                message=result.message,
                parameters=intent.parameters,
            )
            
            # Log interaction to Honcho memory
            if self._session_id and self._memory.config.enabled:
                try:
                    intent_dict = {
                        "tool": intent.tool,
                        "action": intent.action,
                        "parameters": intent.parameters,
                        "confidence": intent.confidence,
                    } if intent else None
                    asyncio.run(
                        self._memory.log_interaction(
                            session_id=self._session_id,
                            user_input=transcript,
                            assistant_response=result.message,
                            intent=intent_dict,
                        )
                    )
                except Exception as e:
                    log.debug("Failed to log interaction to Honcho: %s", e)
            
            if result.success:
                log.info("Command executed: %s", result.message)
                # Handle post-song welcome for play_song command
                if "song" in transcript.lower() and "play" in transcript.lower():
                    threading.Timer(JARVIS_AFTER_SONG_DELAY_S, self._play_welcome).start()
            else:
                log.info("Command failed: %s", result.message)
        else:
            log.warning("Validated intent references unknown tool: %s", intent.tool)
            self._session_logger.log_command(
                transcript=transcript,
                tool=intent.tool,
                action=intent.action,
                latency=time.time() - start_time,
                success=False,
                message=f"Unknown tool: {intent.tool}",
            )
        
        # Reset state after execution
        self._is_processing = False
        self._is_listening = False

    def _handle_confirmation_response(self, transcript: str, stt_latency: float) -> None:
        """Handle user's yes/no response to a confirmation request."""
        if not self._pending_confirmation:
            return
        
        confirmation = classify_confirmation(transcript)
        log.info("Confirmation response: %s -> %s", transcript, confirmation)
        
        pending = self._pending_confirmation
        self._pending_confirmation = None
        
        intent = pending["intent"]
        start_time = pending["start_time"]
        original_transcript = pending["transcript"]
        
        if confirmation == "yes":
            log.info("User confirmed, executing %s.%s", intent.tool, intent.action)
            # Execute the pending command
            cmd = self._commands.get(intent.tool)
            if cmd:
                result = cmd.execute(original_transcript, **intent.parameters)
                exec_latency = time.time() - start_time
                
                self._session_logger.log_command(
                    transcript=original_transcript,
                    tool=intent.tool,
                    action=intent.action,
                    latency=exec_latency,
                    success=result.success,
                    message=result.message,
                    parameters=intent.parameters,
                )
                
                if result.success:
                    log.info("Command executed: %s", result.message)
                    if "song" in original_transcript.lower() and "play" in original_transcript.lower():
                        threading.Timer(JARVIS_AFTER_SONG_DELAY_S, self._play_welcome).start()
                else:
                    log.info("Command failed: %s", result.message)
            else:
                log.warning("Pending intent references unknown tool: %s", intent.tool)
                self._session_logger.log_command(
                    transcript=original_transcript,
                    tool=intent.tool,
                    action=intent.action,
                    latency=time.time() - start_time,
                    success=False,
                    message=f"Unknown tool: {intent.tool}",
                )
        elif confirmation == "no":
            log.info("User denied confirmation for %s.%s", intent.tool, intent.action)
            self._tts.speak("Cancelled.")
            self._session_logger.log_command(
                transcript=original_transcript,
                tool=intent.tool,
                action=intent.action,
                latency=time.time() - start_time,
                success=False,
                message="User denied confirmation",
            )
        else:
            # Unclear response - ask again
            log.info("Unclear confirmation response, asking again: %s", transcript)
            self._tts.speak("I didn't understand. Please say yes or no.")
            # Restore pending confirmation
            self._pending_confirmation = pending
        
        # Reset state
        self._is_processing = False
        self._is_listening = False

    def _classify_with_ai(self, transcript: str) -> IntentClassification | None:
        """Classify transcript using AI intent client with context from memory."""
        if not self._ai_intent.is_available():
            log.debug("AI intent client not available (no API key)")
            return IntentClassification(
                intent=None,
                raw_response="",
                latency=0.0,
                success=False,
                error="OpenRouter API key not configured"
            )
        
        try:
            # Build context package from memory before AI classification
            context_pkg = asyncio.run(
                self._context.build_context(
                    user_message=transcript,
                    session_id=getattr(self, '_session_id', None),
                    include_search=True,
                )
            )
            
            context_str = context_pkg.to_string()
            log.debug("Built context package: %d tokens, %d sections", 
                     context_pkg.total_tokens, len(context_pkg.sections))
            
            # Use sync version since we're in a thread
            return self._ai_intent.classify_sync(transcript, context=context_str)
        except Exception as e:
            log.error("AI intent classification error: %s", e)
            return IntentClassification(
                intent=None,
                raw_response="",
                latency=0.0,
                success=False,
                error=str(e)
            )

    def _play_welcome(self) -> None:
        """Play welcome phrase via TTS."""
        if not JARVIS_WELCOME_ENABLED:
            return
        phrase = JARVIS_WELCOME_PHRASE
        log.info("Playing welcome phrase: %s", phrase[:50])

        tts_start = time.time()
        success = self._tts.speak(phrase)
        tts_latency = time.time() - tts_start

        # Determine which provider was used (first available in preference order)
        provider_used = "unknown"
        for p in self._tts.preference_order:
            eng = self._tts._get_engine(p)
            if eng and eng.is_available:
                provider_used = p
                break

        self._session_logger.log_tts(
            provider=provider_used,
            text=phrase,
            latency=tts_latency,
            success=success,
        )

    def run(self) -> None:
        """Start the main listening loop."""
        log.info("Starting JARVIS...")

        # Play welcome phrase once at startup
        self._play_welcome()

        # Check STT availability
        if not self._stt.is_available():
            log.error("STT engine not available. Exiting.")
            self._session_logger.log_error(
                error_message="STT engine not available",
                error_type="STTUnavailable",
                context={"model": "faster-whisper"}
            )
            return

        # Check available TTS providers
        available_tts = self._tts.available_providers()
        if available_tts:
            log.info("Available TTS providers: %s", ", ".join(available_tts))
        else:
            log.warning("No TTS providers available; responses will be silent.")
            self._session_logger.log_error(
                error_message="No TTS providers available",
                error_type="TTSUnavailable",
                context={"preference_order": self._tts.preference_order}
            )

        # Log AI intent availability
        if self._ai_intent.is_available():
            log.info("AI intent classification enabled (model: %s)", self._ai_intent.model)
        else:
            log.info("AI intent classification disabled (no OpenRouter API key)")

        # Log validation layer availability
        log.info("Validation layer enabled")

        # Initialize Honcho memory
        if self._memory.config.enabled:
            log.info("Initializing Honcho memory (workspace: %s)...", self._memory.config.workspace)
            try:
                success = asyncio.run(self._memory.ensure_workspace())
                if success:
                    # Create/get peer and session
                    self._session_id = asyncio.run(
                        self._memory.create_session(peer_id="user")
                    )
                    if self._session_id:
                        log.info("Honcho session created: %s", self._session_id)
                    else:
                        log.warning("Failed to create Honcho session")
                else:
                    log.warning("Failed to initialize Honcho workspace")
            except Exception as e:
                log.error("Honcho initialization failed: %s", e)

        # Log AI intent availability
        if self._ai_intent.is_available():
            log.info("AI intent classification enabled (model: %s)", self._ai_intent.model)
        else:
            log.info("AI intent classification disabled (no OpenRouter API key)")

        # Log validation layer availability
        log.info("Validation layer enabled")

        # Start wake word detector
        if self._wake_detector:
            if not self._wake_detector.start():
                log.error("Failed to start wake word detector")
                return
        else:
            log.warning("Wake word detector not available (Vosk not configured)")

        self._running = True

        # Keep running - wake word detector runs in background
        log.info("JARVIS ready. Say 'Hey Jarvis' or 'Jarvis' to activate.")

        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error("Runtime error: %s", e)
            self._session_logger.log_error(
                error_message=str(e),
                error_type=type(e).__name__,
                context={"stage": "main_loop"}
            )
            raise
        finally:
            if self._wake_detector:
                self._wake_detector.stop()
            if self._session_id:
                asyncio.run(self._memory.close())

    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False


def main() -> None:
    """Entry point for the application."""
    orchestrator = JarvisOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        log.info("Stopped by user.")
        close_session_logger("user_interrupt")
    except Exception as e:
        log.error("Fatal error: %s", e)
        close_session_logger("fatal_error")
        raise
    finally:
        close_session_logger("normal")