#!/usr/bin/env python3
"""Commands package — registry and built-in commands."""

from .registry import Command, CommandRegistry, CommandResult  # noqa: F401
from .system_commands import (  # noqa: F401
    PlaySongCommand,
)

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandResult",
    "PlaySongCommand",
]