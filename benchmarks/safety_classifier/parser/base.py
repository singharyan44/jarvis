"""Base classes for command parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class OperationType(str, Enum):
    """Types of operations a command can perform."""
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    FILESYSTEM_DELETE = "filesystem_delete"
    PROCESS_SPAWN = "process_spawn"
    NETWORK_READ = "network_read"
    NETWORK_WRITE = "network_write"
    CREDENTIAL_ACCESS = "credential_access"
    PRIVILEGE_CHANGE = "privilege_change"
    PERSISTENCE = "persistence"
    PACKAGE_INSTALL = "package_install"


class ControlFlowType(str, Enum):
    """Control flow types in shell commands."""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PIPELINE = "pipeline"
    BACKGROUND = "background"
    SUBSHELL = "subshell"


@dataclass
class Operation:
    """A single operation extracted from a command."""
    type: OperationType
    target: str
    recursive: bool = False
    details: dict = field(default_factory=dict)
    risk_score: float = 0.0  # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "target": self.target,
            "recursive": self.recursive,
            "details": self.details,
            "risk_score": self.risk_score,
        }


@dataclass
class ParsedCommand:
    """Result of parsing a shell command."""
    shell: str
    command: str
    operations: list[Operation]
    control_flow: list[ControlFlowType]
    processes_spawned: int
    shell_features: list[str]
    raw_ast: Optional[dict] = None
    parse_errors: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return len(self.parse_errors) > 0

    def max_risk_score(self) -> float:
        if not self.operations:
            return 0.0
        return max(op.risk_score for op in self.operations)

    def effect_types(self) -> list[str]:
        return sorted(set(op.type.value for op in self.operations))

    def to_dict(self) -> dict:
        return {
            "shell": self.shell,
            "command": self.command,
            "operations": [op.to_dict() for op in self.operations],
            "control_flow": [cf.value for cf in self.control_flow],
            "processes_spawned": self.processes_spawned,
            "shell_features": self.shell_features,
            "parse_errors": self.parse_errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass
class ParseResult:
    """Result of parsing attempt."""
    success: bool
    parsed: Optional[ParsedCommand] = None
    error: str = ""

    @classmethod
    def ok(cls, parsed: ParsedCommand) -> "ParseResult":
        return cls(success=True, parsed=parsed)

    @classmethod
    def fail(cls, error: str, shell: str = "bash", command: str = "") -> "ParseResult":
        parsed = ParsedCommand(
            shell=shell,
            command=command,
            operations=[],
            control_flow=[],
            processes_spawned=0,
            shell_features=[],
            parse_errors=[error],
        )
        return cls(success=False, parsed=parsed, error=error)