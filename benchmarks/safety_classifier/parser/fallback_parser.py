"""Fallback parser for commands that can't be parsed by specific parsers."""

from __future__ import annotations

import re
import shlex
from typing import Optional

from .base import (
    Operation,
    OperationType,
    ControlFlowType,
    ParsedCommand,
    ParseResult,
)


class FallbackParser:
    """Generic fallback parser using heuristic analysis."""

    # Generic risky patterns
    RISKY_PATTERNS = [
        (r"\brm\s+.*-rf\b", OperationType.FILESYSTEM_DELETE, 0.9),
        (r"\brm\s+.*-r\b", OperationType.FILESYSTEM_DELETE, 0.7),
        (r"\bshred\b", OperationType.FILESYSTEM_DELETE, 0.9),
        (r"\bdd\s+.*of=", OperationType.FILESYSTEM_DELETE, 0.9),
        (r"\bmkfs\b", OperationType.FILESYSTEM_DELETE, 0.9),
        (r">\s*/dev/", OperationType.FILESYSTEM_DELETE, 0.9),
        (r"\bcurl\b.*-X\s+POST", OperationType.NETWORK_WRITE, 0.8),
        (r"\bcurl\b.*-d\s+", OperationType.NETWORK_WRITE, 0.8),
        (r"\bwget\b.*--post", OperationType.NETWORK_WRITE, 0.8),
        (r"\bnc\b.*-l", OperationType.NETWORK_READ, 0.6),
        (r"\bssh\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bsudo\b", OperationType.PRIVILEGE_CHANGE, 0.8),
        (r"\bchmod\s+.*[47][0-9]{3}", OperationType.PRIVILEGE_CHANGE, 0.8),
        (r"\bchown\s+.*root", OperationType.PRIVILEGE_CHANGE, 0.8),
        (r"\bsetcap\b", OperationType.PRIVILEGE_CHANGE, 0.8),
        (r"\beval\s+\$", OperationType.PROCESS_SPAWN, 0.8),
        (r"\$\(", OperationType.PROCESS_SPAWN, 0.7),
        (r"`", OperationType.PROCESS_SPAWN, 0.7),
        (r"\beval\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bpython\s+-c\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bperl\s+-e\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bnode\s+-e\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bbash\s+-c\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\bsh\s+-c\b", OperationType.PROCESS_SPAWN, 0.7),
        (r"\.ssh/id_rsa", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"\.ssh/id_ed25519", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"\.aws/credentials", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"/etc/passwd", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"/etc/shadow", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"/etc/sudoers", OperationType.CREDENTIAL_ACCESS, 0.9),
        (r"\bcp\b", OperationType.FILESYSTEM_WRITE, 0.5),
        (r"\bmv\b", OperationType.FILESYSTEM_WRITE, 0.5),
        (r"\btouch\b", OperationType.FILESYSTEM_WRITE, 0.3),
        (r"\bcat\b", OperationType.FILESYSTEM_READ, 0.3),
        (r"\bgrep\b", OperationType.FILESYSTEM_READ, 0.3),
        (r"\bfind\b", OperationType.FILESYSTEM_READ, 0.4),
    ]

    def __init__(self):
        self._compiled = [(re.compile(p, re.IGNORECASE), t, s) for p, t, s in self.RISKY_PATTERNS]

    def parse(self, command: str, shell: str = "bash") -> ParseResult:
        """Parse using pattern matching."""
        try:
            operations = []
            control_flow = self._detect_control_flow(command)
            features = self._detect_features(command)

            # Match patterns
            for pattern, op_type, risk in self._compiled:
                if pattern.search(command):
                    operations.append(Operation(
                        type=op_type,
                        target="pattern_match",
                        details={"pattern": pattern.pattern},
                        risk_score=risk,
                    ))

            # If no patterns matched, create a generic read operation
            if not operations:
                tokens = shlex.split(command, posix=True)
                if tokens:
                    operations.append(Operation(
                        type=OperationType.FILESYSTEM_READ,
                        target=tokens[0],
                        risk_score=0.3,
                    ))

            parsed = ParsedCommand(
                shell=shell,
                command=command,
                operations=operations,
                control_flow=control_flow,
                processes_spawned=self._count_processes(command),
                shell_features=features,
            )

            return ParseResult.ok(parsed)

        except Exception as e:
            return ParseResult.fail(f"Fallback parse error: {e}", shell, command)

    def _detect_control_flow(self, command: str) -> list:
        flows = []
        if "&&" in command or "||" in command:
            flows.append(ControlFlowType.CONDITIONAL)
        if ";" in command:
            flows.append(ControlFlowType.SEQUENTIAL)
        if "|" in command:
            flows.append(ControlFlowType.PIPELINE)
        if "&" in command and "&&" not in command:
            flows.append(ControlFlowType.BACKGROUND)
        if "$(" in command or "`" in command:
            flows.append(ControlFlowType.SUBSHELL)
        if re.search(r"\b(for|while|until)\b", command):
            flows.append(ControlFlowType.LOOP)
        return flows

    def _detect_features(self, command: str) -> list[str]:
        features = []
        if "&&" in command or "||" in command:
            features.append("chaining")
        if "|" in command:
            features.append("pipeline")
        if "$(" in command or "`" in command:
            features.append("subshell")
        if "${" in command:
            features.append("parameter_expansion")
        if "*" in command or "?" in command or "[" in command:
            features.append("glob")
        if "sudo" in command:
            features.append("sudo")
        return features

    def _count_processes(self, command: str) -> int:
        count = 1
        for pattern in ["bash -c", "sh -c", "python -c", "perl -e", "node -e", "$(", "`", "|"]:
            count += command.count(pattern)
        return max(1, count)


# Singleton
_fallback_parser = FallbackParser()


def parse_fallback(command: str, shell: str = "bash") -> ParseResult:
    """Parse using fallback parser."""
    return _fallback_parser.parse(command, shell)