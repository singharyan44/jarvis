#!/usr/bin/env python3
"""
Command registry and base classes.

Provides a plugin-style command system where each command is a self-contained
class with triggers, an execute method, and metadata.

Phase 3 enhancement: Structured intent output for AI routing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str = ""
    data: Any = None


@dataclass
class Intent:
    """
    Structured intent representation (Phase 3 / Phase 5 ready).
    
    This is what the AI intent layer (Phase 5) will produce.
    For now, the command registry creates this from trigger matches.
    """
    intent_type: str  # "command", "query", "confirmation", etc.
    tool: str         # Command name (e.g., "open_cursor")
    action: str       # Action to perform (e.g., "launch")
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_transcript: str = ""


class Command(ABC):
    """Base class for all commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this command."""
        ...

    @property
    @abstractmethod
    def triggers(self) -> list[str]:
        """Keywords/phrases that activate this command (case-insensitive substring match)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        return ""

    @property
    def default_action(self) -> str:
        """Default action name for this command."""
        return "execute"

    @property
    def valid_actions(self) -> list[str]:
        """List of valid action names for this command. Override to support multiple actions."""
        return [self.default_action, "execute"]

    @abstractmethod
    def execute(self, transcript: str, **kwargs) -> CommandResult:
        """
        Execute the command.

        Args:
            transcript: Full recognized speech text.
            **kwargs: Additional context (e.g., config, services).

        Returns:
            CommandResult indicating success/failure and optional message/data.
        """
        ...

    def to_intent(self, transcript: str, action: str | None = None, **params) -> Intent:
        """Create an Intent from this command match."""
        return Intent(
            intent_type="command",
            tool=self.name,
            action=action or self.default_action,
            parameters=params,
            confidence=1.0,
            raw_transcript=transcript,
        )


class CommandRegistry:
    """Registry for managing and executing commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register a command instance."""
        if command.name in self._commands:
            raise ValueError(f"Command '{command.name}' already registered")
        self._commands[command.name] = command

    def get(self, name: str) -> Command | None:
        """Get a command by name."""
        return self._commands.get(name)

    def find_matching(self, transcript: str) -> Command | None:
        """
        Find the first command whose triggers appear in the transcript.

        Returns the command instance or None if no match.
        """
        lower = transcript.lower()
        for cmd in self._commands.values():
            for trigger in cmd.triggers:
                if trigger.lower() in lower:
                    return cmd
        return None

    def find_matching_with_intent(self, transcript: str) -> tuple[Command | None, Intent | None]:
        """
        Find matching command and return both command and structured intent.

        Returns (command, intent) or (None, None) if no match.
        """
        cmd = self.find_matching(transcript)
        if cmd is None:
            return None, None
        return cmd, cmd.to_intent(transcript)

    def execute_matching(self, transcript: str, **kwargs) -> CommandResult:
        """
        Find and execute the first matching command.

        Returns CommandResult (success=False if no match).
        """
        cmd = self.find_matching(transcript)
        if cmd is None:
            return CommandResult(success=False, message="No matching command")
        return cmd.execute(transcript, **kwargs)

    def execute_intent(self, intent: Intent, **kwargs) -> CommandResult:
        """
        Execute a command by structured intent (for AI routing).

        Args:
            intent: Structured Intent with tool, action, parameters.
            **kwargs: Additional context.

        Returns:
            CommandResult from the command execution.
        """
        cmd = self._commands.get(intent.tool)
        if cmd is None:
            return CommandResult(success=False, message=f"Unknown tool: {intent.tool}")
        # Pass intent parameters to execute
        return cmd.execute(intent.raw_transcript, **intent.parameters, **kwargs)

    def list_commands(self) -> list[Command]:
        """Return all registered commands."""
        return list(self._commands.values())