#!/usr/bin/env python3
"""
System-level commands: play song, execute shell, etc.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

from core.config import SONG_URI
from core.logging import log
from commands.registry import Command, CommandResult


class PlaySongCommand(Command):
    """Play the configured song/media URI."""

    @property
    def name(self) -> str:
        return "play_song"

    @property
    def triggers(self) -> list[str]:
        return ["play song", "play music", "start song", "start music"]

    @property
    def description(self) -> str:
        return "Play the configured song/media"

    def execute(self, transcript: str, **kwargs) -> CommandResult:
        uri = SONG_URI.strip()
        if not uri:
            return CommandResult(success=False, message="No SONG_URI configured")

        try:
            if sys.platform == "win32":
                os.startfile(uri)
            else:
                webbrowser.open(uri)
            log.info("Opened song URI: %s", uri)
            return CommandResult(success=True, message="Playing song")
        except OSError as e:
            log.warning("Could not open SONG_URI: %s", e)
            return CommandResult(success=False, message=f"Could not open song: {e}")


class ExecuteShellCommand(Command):
    """Execute a shell command (requires confirmation)."""

    @property
    def name(self) -> str:
        return "execute_shell"

    @property
    def triggers(self) -> list[str]:
        return ["run command", "execute command", "run shell", "execute shell", "run in terminal"]

    @property
    def description(self) -> str:
        return "Execute a shell command (requires confirmation)"

    @property
    def valid_actions(self) -> list[str]:
        return ["run", "execute"]

    def execute(self, transcript: str, **kwargs) -> CommandResult:
        command = kwargs.get("command", "").strip()
        if not command:
            return CommandResult(success=False, message="No command provided")

        # Safety: reject obviously destructive commands without explicit force
        dangerous_patterns = [
            "rm -rf /", "format", "mkfs", "dd if=", "shutdown", "reboot",
            "poweroff", "halt", "> /dev/", "chmod 777", "chown -R"
        ]
        cmd_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                return CommandResult(
                    success=False, 
                    message=f"Command blocked: contains dangerous pattern '{pattern}'. Use with explicit confirmation."
                )

        try:
            log.info("Executing shell command: %s", command)
            
            # Run with timeout and capture output
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=os.path.expanduser("~")
            )
            
            output = result.stdout.strip()
            error = result.stderr.strip()
            
            if result.returncode == 0:
                msg = f"Command completed (exit code {result.returncode})"
                if output:
                    msg += f"\nOutput: {output[:500]}"
                log.info("Shell command succeeded: %s", command)
                return CommandResult(success=True, message=msg, data={"stdout": output, "stderr": error, "returncode": result.returncode})
            else:
                msg = f"Command failed (exit code {result.returncode})"
                if error:
                    msg += f"\nError: {error[:500]}"
                log.warning("Shell command failed: %s", command)
                return CommandResult(success=False, message=msg, data={"stdout": output, "stderr": error, "returncode": result.returncode})
                
        except subprocess.TimeoutExpired:
            log.error("Shell command timed out: %s", command)
            return CommandResult(success=False, message="Command timed out after 30 seconds")
        except Exception as e:
            log.error("Shell command error: %s", e)
            return CommandResult(success=False, message=f"Execution error: {e}")