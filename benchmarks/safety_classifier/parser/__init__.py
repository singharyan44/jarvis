"""Parser package for shell command analysis."""

from .base import (
    ParsedCommand,
    Operation,
    OperationType,
    ControlFlowType,
    ParseResult,
)
from .bash_parser import BashParser, parse_bash
from .fallback_parser import FallbackParser, parse_fallback

try:
    from .python_parser import PythonParser, parse_python
    HAS_PYTHON_PARSER = True
except ImportError:
    HAS_PYTHON_PARSER = False

try:
    from .powershell_parser import PowerShellParser, parse_powershell
    HAS_POWERSHELL_PARSER = True
except ImportError:
    HAS_POWERSHELL_PARSER = False

__all__ = [
    "ParsedCommand",
    "Operation",
    "OperationType",
    "ControlFlowType",
    "ParseResult",
    "BashParser",
    "FallbackParser",
    "parse_bash",
    "parse_fallback",
    "HAS_PYTHON_PARSER",
    "HAS_POWERSHELL_PARSER",
]

if HAS_PYTHON_PARSER:
    __all__.append("PythonParser")
    __all__.append("parse_python")
if HAS_POWERSHELL_PARSER:
    __all__.append("PowerShellParser")
    __all__.append("parse_powershell")