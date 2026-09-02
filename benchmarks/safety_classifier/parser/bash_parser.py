"""Bash command parser using regex-based analysis."""

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


class BashParser:
    """Parse bash commands into structured operations."""

    # High-risk command patterns
    DESTRUCTIVE_CMDS = {
        "rm", "shred", "dd", "mkfs", "fdisk", "parted",
        "truncate", ">",
    }

    WRITE_CMDS = {
        "cp", "mv", "touch", "echo", "printf", "cat", "tee",
        "sed", "awk", "perl", "python", ">>",
    }

    READ_CMDS = {
        "cat", "less", "more", "head", "tail", "grep", "find",
        "ls", "stat", "file", "md5sum", "sha256sum",
    }

    NETWORK_CMDS = {
        "curl", "wget", "nc", "netcat", "ssh", "scp", "rsync",
        "sftp", "telnet", "nmap",
    }

    PROCESS_CMDS = {
        "bash", "sh", "zsh", "fish", "python", "perl", "node",
        "php", "ruby", "eval", "exec",
    }

    PRIVILEGE_CMDS = {
        "sudo", "su", "doas", "pkexec", "chmod", "chown",
        "setcap", "getcap",
    }

    PERSISTENCE_CMDS = {
        "crontab", "systemctl", "service", "update-rc.d",
        "chkconfig",
    }

    PACKAGE_CMDS = {
        "apt", "apt-get", "yum", "dnf", "pacman", "pip",
        "npm", "cargo", "go", "gem",
    }

    CREDENTIAL_PATHS = [
        r"\.ssh/id_rsa",
        r"\.ssh/id_ed25519",
        r"\.aws/credentials",
        r"\.config/gcloud",
        r"\.docker/config\.json",
        r"\.netrc",
        r"/etc/passwd",
        r"/etc/shadow",
        r"/etc/sudoers",
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for command detection."""
        self.credential_pattern = re.compile(
            r"|".join(self.CREDENTIAL_PATHS), re.IGNORECASE
        )
        self.destructive_pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in self.DESTRUCTIVE_CMDS) + r")\b"
        )
        self.write_pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in self.WRITE_CMDS) + r")\b"
        )
        self.network_pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in self.NETWORK_CMDS) + r")\b"
        )

    def parse(self, command: str, shell: str = "bash") -> ParseResult:
        """Parse a bash command into structured operations."""
        try:
            # Tokenize
            tokens = shlex.split(command, posix=True)
            if not tokens:
                return ParseResult.fail("Empty command", shell, command)

            # Detect control flow
            control_flow = self._detect_control_flow(command)

            # Extract operations
            operations = self._extract_operations(tokens, command)

            # Count processes
            processes = self._count_processes(command)

            # Detect shell features
            features = self._detect_features(command)

            parsed = ParsedCommand(
                shell="bash",
                command=command,
                operations=operations,
                control_flow=control_flow,
                processes_spawned=processes,
                shell_features=features,
            )

            return ParseResult.ok(parsed)

        except Exception as e:
            return ParseResult.fail(f"Parse error: {e}", shell, command)

    def _detect_control_flow(self, command: str) -> list:
        """Detect control flow constructs."""
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
        if re.search(r"\$\(|`", command):
            flows.append(ControlFlowType.SUBSHELL)
        if re.search(r"(for|while|until)\s", command):
            flows.append(ControlFlowType.LOOP)
        return flows

    def _extract_operations(self, tokens: list, raw: str) -> list[Operation]:
        """Extract operations from tokenized command."""
        operations = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Skip control operators
            if token in ["&&", "||", ";", "|", "&", "(", ")"]:
                i += 1
                continue

            # Check for redirections
            if token in [">", ">>", "<", "<<", "2>", "2>>", "&>"]:
                i += 1
                if i < len(tokens):
                    i += 1
                continue

            # Main command
            if i == 0 or tokens[i-1] in ["&&", "||", ";", "|", "&", "("]:
                ops = self._analyze_command(token, tokens[i+1:] if i+1 < len(tokens) else [], raw)
                operations.extend(ops)

            i += 1

        return operations

    def _analyze_command(self, cmd: str, args: list, raw: str) -> list[Operation]:
        """Analyze a single command and its arguments."""
        operations = []

        # Destructive commands
        if cmd in self.DESTRUCTIVE_CMDS or (cmd == "find" and "-delete" in args):
            operations.append(Operation(
                type=OperationType.FILESYSTEM_DELETE,
                target=self._extract_target(args),
                recursive="-r" in args or "-rf" in args or "-R" in args,
                details={"command": cmd, "args": args},
                risk_score=0.9,
            ))

        # Write commands
        elif cmd in self.WRITE_CMDS:
            target = self._extract_target(args)
            risk = 0.7 if cmd in ["cp", "mv", "sed", "awk"] else 0.5
            if self._is_credential_target(target):
                operations.append(Operation(
                    type=OperationType.CREDENTIAL_ACCESS,
                    target=target,
                    details={"command": cmd},
                    risk_score=0.9,
                ))
            else:
                operations.append(Operation(
                    type=OperationType.FILESYSTEM_WRITE,
                    target=target,
                    details={"command": cmd},
                    risk_score=risk,
                ))

        # Read commands
        elif cmd in self.READ_CMDS:
            target = self._extract_target(args)
            risk = 0.6 if self._is_credential_target(target) else 0.3
            operations.append(Operation(
                type=OperationType.CREDENTIAL_ACCESS if self._is_credential_target(target)
                else OperationType.FILESYSTEM_READ,
                target=target,
                details={"command": cmd},
                risk_score=risk,
            ))

        # Network commands
        elif cmd in self.NETWORK_CMDS:
            op_type = OperationType.NETWORK_WRITE if cmd in ["curl", "wget", "nc", "scp", "rsync"] else OperationType.NETWORK_READ
            risk = 0.8 if "-X POST" in " ".join(args) or "-d" in args else 0.6
            operations.append(Operation(
                type=op_type,
                target=self._extract_target(args),
                details={"command": cmd, "args": args},
                risk_score=risk,
            ))

        # Process spawning
        elif cmd in self.PROCESS_CMDS:
            operations.append(Operation(
                type=OperationType.PROCESS_SPAWN,
                target=cmd,
                details={"args": args},
                risk_score=0.7,
            ))

        # Privilege commands
        elif cmd in self.PRIVILEGE_CMDS:
            operations.append(Operation(
                type=OperationType.PRIVILEGE_CHANGE,
                target=self._extract_target(args),
                details={"command": cmd},
                risk_score=0.8,
            ))

        # Persistence
        elif cmd in self.PERSISTENCE_CMDS:
            operations.append(Operation(
                type=OperationType.PERSISTENCE,
                target=self._extract_target(args),
                details={"command": cmd},
                risk_score=0.7,
            ))

        # Package install
        elif cmd in self.PACKAGE_CMDS:
            operations.append(Operation(
                type=OperationType.PACKAGE_INSTALL,
                target=self._extract_target(args),
                details={"command": cmd},
                risk_score=0.6,
            ))

        # Default: unknown command
        else:
            operations.append(Operation(
                type=OperationType.FILESYSTEM_READ,
                target=cmd,
                details={"command": cmd, "args": args, "unknown": True},
                risk_score=0.4,
            ))

        return operations

    def _extract_target(self, args: list) -> str:
        """Extract target path from arguments."""
        for arg in args:
            if not arg.startswith("-") and not arg.startswith("$"):
                return arg
        return "unknown"

    def _is_credential_target(self, target: str) -> bool:
        """Check if target is a credential file."""
        return any(re.search(p, target, re.IGNORECASE) for p in self.CREDENTIAL_PATHS)

    def _count_processes(self, command: str) -> int:
        """Count number of processes spawned."""
        count = 1
        count += command.count("bash -c")
        count += command.count("sh -c")
        count += command.count("python -c")
        count += command.count("perl -e")
        count += command.count("node -e")
        count += command.count("$( ")
        count += command.count("`")
        count += len(re.findall(r"\|\s*\w", command))
        return max(1, count)

    def _detect_features(self, command: str) -> list[str]:
        """Detect shell features used."""
        features = []
        if "&&" in command or "||" in command:
            features.append("chaining")
        if "|" in command:
            features.append("pipeline")
        if re.search(r"\$\(", command) or "`" in command:
            features.append("subshell")
        if re.search(r"\$\{", command):
            features.append("parameter_expansion")
        if re.search(r"\*\*|\*\?|\[", command):
            features.append("glob")
        if "sudo" in command:
            features.append("sudo")
        if re.search(r">\s*/dev/", command):
            features.append("device_write")
        return features


# Singleton instance
_bash_parser = BashParser()


def parse_bash(command: str) -> ParseResult:
    """Parse a bash command."""
    return _bash_parser.parse(command)