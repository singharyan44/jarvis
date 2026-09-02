#!/usr/bin/env python3
"""
Validation Layer — Safety checks for AI-generated intents before execution.

Ensures AI output is safe to execute by validating:
- Tool exists in registry
- Action exists for the tool
- Parameters are valid (type, required fields)
- Dangerous actions require confirmation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from commands.registry import CommandRegistry, Intent
from core.logging import log


@dataclass
class ValidationResult:
    """Result of intent validation."""
    valid: bool
    intent: Intent | None = None
    error: str = ""
    requires_confirmation: bool = False
    confirmation_message: str = ""


class ActionValidator:
    """
    Validates structured intents against registered commands.
    
    Usage:
        validator = ActionValidator(command_registry)
        result = validator.validate(intent)
        if result.valid:
            if result.requires_confirmation:
                # Ask user for confirmation
                pass
            # Execute intent
        else:
            # Handle validation error
            pass
    """
    
    # Actions that require explicit user confirmation
    DANGEROUS_ACTIONS = {
        "shutdown",
        "restart",
        "reboot",
        "delete",
        "remove",
        "uninstall",
        "format",
        "wipe",
        "kill",
        "terminate",
    }
    
    # Tools that have dangerous actions
    DANGEROUS_TOOLS = {
        "system",
        "power",
        "disk",
        "network",
        "execute_shell",
    }
    
    def __init__(self, command_registry: CommandRegistry) -> None:
        self.registry = command_registry
    
    def validate(self, intent: Intent) -> ValidationResult:
        """
        Validate an intent against the command registry.
        
        Args:
            intent: Structured intent from AI or keyword matching
            
        Returns:
            ValidationResult with validation outcome
        """
        # 1. Check tool exists
        cmd = self.registry.get(intent.tool)
        if cmd is None:
            return ValidationResult(
                valid=False,
                error=f"Unknown tool: '{intent.tool}'. Available tools: {list(self.registry._commands.keys())}"
            )
        
        # 2. Check action exists for this tool
        if not self._action_exists(cmd, intent.action):
            return ValidationResult(
                valid=False,
                error=f"Action '{intent.action}' not supported by tool '{intent.tool}'. Expected: '{cmd.default_action}'"
            )
        
        # 3. Validate parameters
        param_error = self._validate_parameters(cmd, intent.parameters)
        if param_error:
            return ValidationResult(
                valid=False,
                error=f"Invalid parameters for {intent.tool}.{intent.action}: {param_error}"
            )
        
        # 4. Check if confirmation required
        requires_confirmation, conf_message = self._check_confirmation_required(intent)
        
        return ValidationResult(
            valid=True,
            intent=intent,
            requires_confirmation=requires_confirmation,
            confirmation_message=conf_message,
        )
    
    def _action_exists(self, cmd, action: str) -> bool:
        """Check if action is valid for the command."""
        return action in cmd.valid_actions
    
    def _validate_parameters(self, cmd, parameters: dict[str, Any]) -> str | None:
        """
        Validate parameters against command expectations.
        
        Returns error message if invalid, None if valid.
        """
        # Get command's expected parameters from its execute signature
        # For now, we accept any parameters (commands handle their own validation)
        # Could be extended with JSON schema validation
        return None
    
    def _check_confirmation_required(self, intent: Intent) -> tuple[bool, str]:
        """
        Check if this intent requires user confirmation.
        
        Returns (requires_confirmation, confirmation_message)
        """
        tool_lower = intent.tool.lower()
        action_lower = intent.action.lower()
        
        # Check for dangerous actions
        if action_lower in self.DANGEROUS_ACTIONS:
            return True, f"⚠️ This will {action_lower} the system. Are you sure?"
        
        # Check for dangerous tools with any action
        if tool_lower in self.DANGEROUS_TOOLS:
            return True, f"⚠️ This will execute a {tool_lower} command. Continue?"
        
        # Check for shutdown-like patterns in tool/action
        if "shutdown" in f"{tool_lower} {action_lower}":
            return True, "⚠️ This will shut down the system. Are you sure?"
        
        return False, ""


class ConfirmationManager:
    """
    Manages user confirmation for dangerous actions.
    
    In a voice assistant context, this would use TTS to ask and STT to listen.
    For now, provides a synchronous interface for testing.
    """
    
    def __init__(self, tts_manager=None) -> None:
        self.tts_manager = tts_manager
    
    def request_confirmation(self, message: str, timeout: float = 10.0) -> bool:
        """
        Request user confirmation for a dangerous action.
        
        Args:
            message: Confirmation message to present to user
            timeout: Seconds to wait for response
            
        Returns:
            True if user confirms, False otherwise
        """
        # In a real implementation, this would:
        # 1. Use TTS to speak the confirmation message
        # 2. Listen for voice response (yes/no)
        # 3. Return based on user response
        
        # For now, log and return False (safe default)
        log.warning("CONFIRMATION REQUIRED: %s", message)
        log.warning("Auto-denying for safety (no voice confirmation implemented)")
        return False
    
    async def request_confirmation_async(self, message: str, timeout: float = 10.0) -> bool:
        """Async version for integration with async pipelines."""
        return self.request_confirmation(message, timeout)