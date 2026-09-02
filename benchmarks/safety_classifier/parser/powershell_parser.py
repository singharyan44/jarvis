"""PowerShell command parser using regex-based analysis."""

from __future__ import annotations

import re
from typing import Optional

from .base import (
    Operation,
    OperationType,
    ControlFlowType,
    ParsedCommand,
    ParseResult,
)


class PowerShellParser:
    """Parse PowerShell commands using regex-based analysis."""

    DESTRUCTIVE_CMDS = {
        "remove-item", "ri", "del", "erase", "rd", "rmdir",
        "clear-content", "clc", "format-volume", "initialize-disk",
    }

    WRITE_CMDS = {
        "set-content", "sc", "add-content", "ac",
        "new-item", "ni", "mkdir", "md",
        "copy-item", "cp", "copy", "move-item", "mv", "move",
        "rename-item", "rn", "ren",
    }

    READ_CMDS = {
        "get-content", "gc", "cat", "type",
        "get-childitem", "gci", "ls", "dir",
        "get-item", "gi",
    }

    NETWORK_CMDS = {
        "invoke-webrequest", "iwr", "curl", "wget",
        "invoke-restmethod", "irm",
        "send-mailmessage",
        "net.tcpclient", "system.net.sockets.tcpclient",
    }

    PROCESS_CMDS = {
        "start-process", "saps", "ps",
        "invoke-expression", "iex",
        "invoke-command", "icm",
        "start-job", "sajb",
    }

    PRIVILEGE_CMDS = {
        "runas", "enable-privilege",
        "set-acl", "icacls", "cacls", "takeown",
    }

    PERSISTENCE_CMDS = {
        "register-scheduledtask", "new-scheduledtask",
        "set-service", "new-service",
        "register-wmievent",
    }

    CREDENTIAL_PATHS = [
        r"\\.ssh\\id_rsa",
        r"\\.aws\\credentials",
        r"\\.config\\gcloud",
        r"\\.docker\\config\.json",
        r"env:",
        r"\$env:",
        r"get-credential",
        r"convertto-securestring",
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self.credential_pattern = re.compile(
            r"|".join(self.CREDENTIAL_PATHS), re.IGNORECASE
        )

    def parse(self, command: str, shell: str = "powershell") -> ParseResult:
        """Parse a PowerShell command."""
        try:
            control_flow = self._detect_control_flow(command)
            operations = self._extract_operations(command)
            features = self._detect_features(command)
            processes = self._count_processes(command)

            parsed = ParsedCommand(
                shell="powershell",
                command=command,
                operations=operations,
                control_flow=control_flow,
                processes_spawned=processes,
                shell_features=features,
            )

            return ParseResult.ok(parsed)

        except Exception as e:
            return ParseResult.fail(f"PowerShell parse error: {e}", shell, command)

    def _detect_control_flow(self, command: str) -> list:
        flows = []
        if re.search(r"\|\|", command):
            flows.append(ControlFlowType.CONDITIONAL)
        if re.search(r"&&", command):
            flows.append(ControlFlowType.CONDITIONAL)
        if re.search(r";", command):
            flows.append(ControlFlowType.SEQUENTIAL)
        if re.search(r"\|", command):
            flows.append(ControlFlowType.PIPELINE)
        if re.search(r"&(?!\&)", command):
            flows.append(ControlFlowType.BACKGROUND)
        if re.search(r"\$\(", command):
            flows.append(ControlFlowType.SUBSHELL)
        if re.search(r"\b(if|foreach|for|while|do|switch)\b", command, re.IGNORECASE):
            flows.append(ControlFlowType.CONDITIONAL)
        if re.search(r"\b(foreach|for|while|do)\b", command, re.IGNORECASE):
            flows.append(ControlFlowType.LOOP)
        return list(set(flows))

    def _extract_operations(self, command: str) -> list:
        operations = []

        # Check destructive
        for cmd in self.DESTRUCTIVE_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                operations.append(Operation(
                    type=OperationType.FILESYSTEM_DELETE,
                    target="filesystem",
                    risk_score=0.9,
                ))

        # Check write
        for cmd in self.WRITE_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                operations.append(Operation(
                    type=OperationType.FILESYSTEM_WRITE,
                    target="filesystem",
                    risk_score=0.6,
                ))

        # Check read
        for cmd in self.READ_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                risk = 0.6 if any(re.search(p, command, re.IGNORECASE) for p in self.CREDENTIAL_PATHS) else 0.3
                operations.append(Operation(
                    type=OperationType.CREDENTIAL_ACCESS if risk > 0.5 else OperationType.FILESYSTEM_READ,
                    target="filesystem",
                    risk_score=risk,
                ))

        # Network
        for cmd in self.NETWORK_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                risk = 0.8 if "-method post" in command.lower() or "-body" in command.lower() else 0.6
                operations.append(Operation(
                    type=OperationType.NETWORK_WRITE,
                    target="network",
                    risk_score=risk,
                ))

        # Process
        for cmd in self.PROCESS_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                operations.append(Operation(
                    type=OperationType.PROCESS_SPAWN,
                    target="process",
                    risk_score=0.7,
                ))

        # Privilege
        for cmd in self.PRIVILEGE_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                operations.append(Operation(
                    type=OperationType.PRIVILEGE_CHANGE,
                    target="privilege",
                    risk_score=0.8,
                ))

        # Persistence
        for cmd in self.PERSISTENCE_CMDS:
            if re.search(rf"\b{re.escape(cmd)}\b", command, re.IGNORECASE):
                operations.append(Operation(
                    type=OperationType.PERSISTENCE,
                    target="persistence",
                    risk_score=0.7,
                ))

        return operations

    def _detect_features(self, command: str) -> list:
        features = []
        if "&&" in command or "||" in command:
            features.append("chaining")
        if "|" in command:
            features.append("pipeline")
        if "$(" in command:
            features.append("subexpression")
        if "${" in command:
            features.append("variable")
        if "*" in command or "?" in command:
            features.append("wildcard")
        if re.search(r"\bsudo\b", command, re.IGNORECASE):
            features.append("elevation")
        return list(set(features))

    def _count_processes(self, command: str) -> int:
        count = 1
        for p in ["start-process", "invoke-expression", "invoke-command", "&", "$("]:
            count += command.lower().count(p.lower())
        return max(1, count)


_powershell_parser = PowerShellParser()


def parse_powershell(command: str) -> ParseResult:
    """Parse a PowerShell command."""
    return _powershell_parser.parse(command)