"""Python command parser using AST analysis."""

from __future__ import annotations

import ast
import shlex
from typing import Optional

from .base import (
    Operation,
    OperationType,
    ControlFlowType,
    ParsedCommand,
    ParseResult,
)


class PythonParser:
    """Parse Python commands using AST."""

    RISKY_MODULES = {
        "os": {"system", "popen", "spawn", "exec"},
        "subprocess": {"run", "call", "Popen", "check_output"},
        "sys": {"exit"},
        "builtins": {"eval", "exec", "compile"},
        "importlib": {"import_module"},
        "__builtin__": {"eval", "exec"},
    }

    RISKY_FUNCTIONS = {
        "eval", "exec", "compile", "__import__",
        "open", "input", "getattr", "setattr",
    }

    NETWORK_MODULES = {
        "urllib", "urllib.request", "urllib.parse",
        "http.client", "http.server",
        "socket", "ssl",
        "requests", "httpx", "aiohttp",
    }

    FILESYSTEM_MODULES = {
        "os", "pathlib", "shutil", "glob", "fnmatch",
        "tempfile", "io",
    }

    def __init__(self):
        pass

    def parse(self, command: str, shell: str = "python") -> ParseResult:
        """Parse a Python command using AST."""
        try:
            # Handle -c flag
            if command.startswith("python -c "):
                code = command[len("python -c "):]
            elif command.startswith("python3 -c "):
                code = command[len("python3 -c "):]
            else:
                code = command

            # Remove surrounding quotes
            code = code.strip()
            if (code.startswith('"') and code.endswith('"')) or \
               (code.startswith("'") and code.endswith("'")):
                code = code[1:-1]

            # Parse AST
            tree = ast.parse(code)

            # Analyze
            operations = self._analyze_ast(tree)
            control_flow = self._detect_control_flow(tree)
            features = self._detect_features(tree)

            parsed = ParsedCommand(
                shell="python",
                command=command,
                operations=operations,
                control_flow=control_flow,
                processes_spawned=1,
                shell_features=features,
            )

            return ParseResult.ok(parsed)

        except SyntaxError as e:
            return ParseResult.fail(f"Python syntax error: {e}", "python", command)
        except Exception as e:
            return ParseResult.fail(f"Python parse error: {e}", "python", command)

    def _analyze_ast(self, tree: ast.AST) -> list:
        """Analyze AST for risky operations."""
        operations = []

        for node in ast.walk(tree):
            # Import analysis
            if isinstance(node, ast.Import):
                for alias in node.names:
                    ops = self._analyze_import(alias.name)
                    operations.extend(ops)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    ops = self._analyze_import(node.module, [a.name for a in node.names])
                    operations.extend(ops)

            # Function calls
            elif isinstance(node, ast.Call):
                ops = self._analyze_call(node)
                operations.extend(ops)

            # Attribute access
            elif isinstance(node, ast.Attribute):
                ops = self._analyze_attribute(node)
                operations.extend(ops)

        # Deduplicate
        seen = set()
        unique = []
        for op in operations:
            key = (op.type, op.target)
            if key not in seen:
                seen.add(key)
                unique.append(op)

        return unique

    def _analyze_import(self, module: str, names: list = None) -> list:
        """Analyze import statements."""
        operations = []

        # Check risky modules
        for mod, funcs in self.RISKY_MODULES.items():
            if module == mod or module.startswith(mod + "."):
                operations.append(Operation(
                    type=OperationType.PROCESS_SPAWN,
                    target=f"import:{module}",
                    details={"module": module, "functions": list(funcs)},
                    risk_score=0.7,
                ))

        # Network modules
        for mod in self.NETWORK_MODULES:
            if module == mod or module.startswith(mod + "."):
                operations.append(Operation(
                    type=OperationType.NETWORK_WRITE,
                    target=f"import:{module}",
                    details={"module": module},
                    risk_score=0.6,
                ))

        # Filesystem modules
        for mod in self.FILESYSTEM_MODULES:
            if module == mod or module.startswith(mod + "."):
                operations.append(Operation(
                    type=OperationType.FILESYSTEM_WRITE,
                    target=f"import:{module}",
                    details={"module": module},
                    risk_score=0.5,
                ))

        return operations

    def _analyze_call(self, node: ast.Call) -> list:
        """Analyze function calls."""
        operations = []

        # Get function name
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check risky functions
        if func_name in self.RISKY_FUNCTIONS:
            return [Operation(
                type=OperationType.PROCESS_SPAWN,
                target=f"call:{func_name}",
                risk_score=0.8,
            )]

        # subprocess calls
        if func_name in ["run", "call", "Popen", "check_output", "system", "popen"]:
            return [Operation(
                type=OperationType.PROCESS_SPAWN,
                target=f"call:{func_name}",
                risk_score=0.8,
            )]

        # os functions
        if func_name in ["system", "popen", "spawn", "execl", "execle", "execlp", "execlpe",
                         "execv", "execve", "execvp", "execvpe", "fork", "forkpty"]:
            return [Operation(
                type=OperationType.PROCESS_SPAWN,
                target=f"os.{func_name}",
                risk_score=0.9,
            )]

        # File operations
        if func_name in ["open", "remove", "unlink", "rmdir", "mkdir", "makedirs",
                         "rename", "replace", "chmod", "chown", "remove"]:
            return [Operation(
                type=OperationType.FILESYSTEM_WRITE,
                target=f"call:{func_name}",
                risk_score=0.6,
            )]

        # Network calls
        if func_name in ["urlopen", "urlretrieve", "Request", "get", "post", "put", "delete"]:
            return [Operation(
                type=OperationType.NETWORK_WRITE,
                target=f"call:{func_name}",
                risk_score=0.7,
            )]

        return []

    def _analyze_attribute(self, node: ast.Attribute) -> list:
        """Analyze attribute access."""
        operations = []

        # os.system, subprocess.run, etc.
        if isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in ["system", "popen", "spawn"]:
                return [Operation(
                    type=OperationType.PROCESS_SPAWN,
                    target=f"os.{node.attr}",
                    risk_score=0.9,
                )]
            if node.value.id == "subprocess" and node.attr in ["run", "call", "Popen"]:
                return [Operation(
                    type=OperationType.PROCESS_SPAWN,
                    target=f"subprocess.{node.attr}",
                    risk_score=0.8,
                )]

        return operations

    def _detect_control_flow(self, tree: ast.AST) -> list:
        flows = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                flows.append(ControlFlowType.CONDITIONAL)
            elif isinstance(node, (ast.For, ast.While)):
                flows.append(ControlFlowType.LOOP)
            elif isinstance(node, ast.Try):
                flows.append(ControlFlowType.CONDITIONAL)
        return list(set(flows))

    def _detect_features(self, tree: ast.AST) -> list:
        features = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile"]:
                        features.append("dynamic_code")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["system", "popen", "run", "Popen"]:
                        features.append("process_spawn")
            elif isinstance(node, ast.Import):
                features.append("imports")
        return list(set(features))


_python_parser = PythonParser()


def parse_python(command: str) -> ParseResult:
    """Parse a Python command."""
    return _python_parser.parse(command)