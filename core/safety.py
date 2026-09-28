#!/usr/bin/env python3
"""
Safety Analyzer — Terminal/Tool-call Authorization Pipeline.

Integrates:
1. Parser (AST/regex) → structured operations
2. Effect Extraction Model (ONNX) → effects + risk score
3. Authorization Model (ONNX) → ALLOW/CONFIRM/DENY/UNCERTAIN
4. Calibrated policy thresholds
5. Policy engine with UNCERTAIN→CONFIRM mapping
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional imports - loaded lazily
_ORT_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None

from benchmarks.safety_classifier.parser import (
    BashParser,
    FallbackParser,
    ParsedCommand,
    Operation,
    OperationType,
    ControlFlowType,
    ParseResult,
    parse_bash,
    parse_fallback,
)

try:
    from benchmarks.safety_classifier.parser import PythonParser, parse_python
    HAS_PYTHON_PARSER = True
except ImportError:
    HAS_PYTHON_PARSER = False

try:
    from benchmarks.safety_classifier.parser import PowerShellParser, parse_powershell
    HAS_POWERSHELL_PARSER = True
except ImportError:
    HAS_POWERSHELL_PARSER = False


class SafetyDecision(str, Enum):
    """Safety decision outcomes."""
    ALLOW = "ALLOW"
    CONFIRM = "CONFIRM"
    DENY = "DENY"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SafetyDecisionResult:
    """Result of safety analysis."""
    decision: SafetyDecision
    confidence: float
    effects: list[str]
    risk_score: float
    parsed_command: Optional[Any] = None
    raw_logits: Optional[list] = None
    calibrated_probs: Optional[list] = None
    explanation: str = ""


@dataclass
class ToolRequest:
    """Structured tool request from the main agent."""
    intent: str
    tool: str
    shell: str
    command: str
    context: dict[str, Any]

    # Parsed by safety analyzer
    parsed: Optional[Any] = None
    effects: Optional[list[str]] = None
    risk_score: Optional[float] = None


class ONNXModel:
    """Wrapper for ONNX model inference."""

    def __init__(self, model_path: str, tokenizer_name: str):
        if not _ORT_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("onnxruntime and transformers required for ONNXModel")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

        self.input_names = ["input_ids", "attention_mask"]
        self.output_name = "logits"

    def predict(self, text: str) -> np.ndarray:
        """Run inference on a single text input."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        return logits[0]  # Return single sample logits

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        """Run inference on a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(None, ort_inputs)[0]
        return logits


class SafetyAnalyzer:
    """
    Terminal/Tool-call Safety Analyzer.

    Pipeline:
    1. Parse command → structured operations
    2. Effect Extraction Model → effects + risk score
    3. Authorization Model → logits
    4. Temperature scaling → calibrated probabilities
    5. Policy engine → ALLOW/CONFIRM/DENY/UNCERTAIN
    """

    EFFECT_LABELS = [
        "filesystem_read",
        "filesystem_write",
        "filesystem_delete",
        "process_spawn",
        "network_read",
        "network_write",
        "credential_access",
        "privilege_change",
        "persistence",
        "package_install",
    ]

    AUTH_LABELS = ["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]
    AUTH_LABEL2ID = {label: i for i, label in enumerate(AUTH_LABELS)}
    AUTH_ID2LABEL = {i: label for i, label in enumerate(AUTH_LABELS)}

    def __init__(
        self,
        model_dir: str,
        parser_shell: str = "bash",
    ):
        """
        Initialize SafetyAnalyzer.

        Args:
            model_dir: Directory containing ONNX models and calibration.json
            parser_shell: Default shell for parsing ("bash", "powershell", "python")
        """
        self.model_dir = Path(model_dir)

        # Load calibration
        cal_path = self.model_dir / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            self.temperature = cal.get("temperature", 1.0)
            self.thresholds = cal.get("thresholds", {
                "allow": 0.999,
                "deny": 0.5,
            })
        else:
            self.temperature = 1.0
            self.thresholds = {"allow": 0.999, "deny": 0.5}

        # Load models (gracefully handle missing dependencies)
        self.effect_model = None
        self.auth_model = None
        self.tokenizer_name = "distilbert-base-uncased"

        effect_path = self.model_dir / "effect_model.onnx"
        auth_path = self.model_dir / "auth_model.onnx"

        if effect_path.exists():
            try:
                self.effect_model = ONNXModel(str(effect_path), self.tokenizer_name)
            except RuntimeError:
                pass  # Dependencies not available
        if auth_path.exists():
            try:
                self.auth_model = ONNXModel(str(auth_path), self.tokenizer_name)
            except RuntimeError:
                pass

        # Parsers
        self.parser_shell = parser_shell
        self.bash_parser = BashParser()
        self.fallback_parser = FallbackParser()
        if HAS_PYTHON_PARSER:
            self.python_parser = PythonParser()
        if HAS_POWERSHELL_PARSER:
            self.powershell_parser = PowerShellParser()

    def analyze(self, tool_request: ToolRequest) -> SafetyDecisionResult:
        """
        Analyze a tool request for safety.

        Returns SafetyDecisionResult with decision and metadata.
        """
        # 1. Parse command
        parsed = self._parse_command(tool_request.command, tool_request.shell)
        tool_request.parsed = parsed

        if parsed.has_errors():
            return SafetyDecisionResult(
                decision=SafetyDecision.CONFIRM,
                confidence=0.0,
                effects=[],
                risk_score=1.0,
                parsed_command=parsed,
                explanation=f"Parse failed: {parsed.parse_errors}",
            )

        # 2. Extract effects
        effects, risk_score = self._extract_effects(parsed, tool_request.context)
        tool_request.effects = effects
        tool_request.risk_score = risk_score

        # 3. Authorization
        decision, confidence, logits, probs = self._authorize(effects, tool_request)

        # 4. Policy
        final_decision = self._apply_policy(decision, confidence)

        return SafetyDecisionResult(
            decision=final_decision,
            confidence=confidence,
            effects=effects,
            risk_score=risk_score,
            parsed_command=parsed,
            raw_logits=logits.tolist() if logits is not None else None,
            calibrated_probs=probs.tolist() if probs is not None else None,
            explanation=self._explanation(final_decision, effects, risk_score),
        )

    def _parse_command(self, command: str, shell: str) -> ParseResult:
        """Parse command using appropriate parser."""
        if shell == "bash":
            return parse_bash(command)
        elif shell == "powershell" and HAS_POWERSHELL_PARSER:
            return parse_powershell(command)
        elif shell == "python" and HAS_PYTHON_PARSER:
            return parse_python(command)
        else:
            return parse_fallback(command, shell)

    def _serialize_for_model(self, tool_request: ToolRequest, parsed: Optional[ParsedCommand] = None) -> str:
        """Serialize tool request for model input."""
        parts = [
            f"intent: {tool_request.intent}",
            f"tool: {tool_request.tool}",
            f"shell: {tool_request.shell}",
            f"command: {tool_request.command}",
        ]

        if parsed and parsed.operations:
            ops = []
            for op in parsed.operations:
                op_str = f"{op.type.value}:{op.target}"
                if op.recursive:
                    op_str += " (recursive)"
                ops.append(op_str)
            parts.append(f"operations: {', '.join(ops)}")

            if parsed.control_flow:
                cf = [cf.value for cf in parsed.control_flow]
                parts.append(f"control_flow: {', '.join(cf)}")

        ctx = tool_request.context
        if ctx:
            ctx_parts = [f"{k}: {v}" for k, v in ctx.items()]
            parts.append(f"context: {', '.join(ctx_parts)}")

        return " [SEP] ".join(parts)

    def _extract_effects(self, parsed: Optional[ParsedCommand], context: dict) -> tuple[list[str], float]:
        """Extract effects using effect extraction model."""
        if not self.effect_model:
            # Fallback: derive from parsed operations
            return self._fallback_effects(parsed)

        # Serialize for effect model
        text = self._serialize_for_model(ToolRequest(
            intent="", tool="", shell="", command="", context=context
        ))

        logits = self.effect_model.predict(text)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid for multi-label

        # Threshold at 0.5
        predicted = (probs > 0.5).astype(int)
        effects = [self.EFFECT_LABELS[i] for i in range(len(self.EFFECT_LABELS)) if predicted[i]]

        # Risk score = max probability of dangerous effects
        dangerous_indices = [2, 3, 6, 7]  # fs_delete, proc_spawn, cred_access, priv_change
        risk_score = float(np.max(probs[dangerous_indices])) if len(probs) > 0 else 0.0

        return effects, risk_score

    def _fallback_effects(self, parsed: Optional[ParsedCommand]) -> tuple[list[str], float]:
        """Fallback effect extraction from parsed operations."""
        if not parsed or not parsed.operations:
            return [], 0.0

        effects = set()
        max_risk = 0.0

        for op in parsed.operations:
            effects.add(op.type.value)
            max_risk = max(max_risk, op.risk_score)

        # Map operations to effect labels
        mapped = []
        for eff in effects:
            if eff in self.EFFECT_LABELS:
                mapped.append(eff)

        return list(mapped), max_risk

    def _authorize(self, effects: list[str], context: dict) -> tuple[SafetyDecision, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run authorization model."""
        if not self.auth_model:
            # Fallback: rule-based
            return self._fallback_authorize(effects, context)

        # Serialize for auth model
        # Build a synthetic tool request for serialization
        tool_request = ToolRequest(
            intent="", tool="", shell="bash", command="", context=context
        )
        text = self._serialize_for_model(tool_request)

        logits = self.auth_model.predict(text)

        # Temperature scaling
        scaled_logits = logits / self.temperature
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        decision = SafetyDecision(self.AUTH_ID2LABEL[pred_idx])

        return decision, confidence, logits, probs

    def _fallback_authorize(self, effects: list[str], context: dict) -> tuple[SafetyDecision, float, None, None]:
        """Rule-based fallback authorization."""
        # Check for dangerous effects
        dangerous = {"filesystem_delete", "process_spawn", "credential_access", "privilege_change", "network_write"}
        high_risk = {"persistence", "package_install", "network_read"}

        has_dangerous = any(e in dangerous for e in effects)
        has_high_risk = any(e in high_risk for e in effects)

        # Check context
        is_root = context.get("privileges") == "root"
        has_network = context.get("network", False)
        is_isolated = context.get("sandbox") == "isolated_docker"

        if has_dangerous and (is_root or has_network or not is_isolated):
            return SafetyDecision.DENY, 0.9, None, None
        elif has_dangerous or has_high_risk:
            return SafetyDecision.CONFIRM, 0.7, None, None
        else:
            return SafetyDecision.ALLOW, 0.8, None, None

    def _apply_policy(self, decision: SafetyDecision, confidence: float) -> SafetyDecision:
        """Apply calibrated policy thresholds."""
        # Map UNCERTAIN to CONFIRM
        if decision == SafetyDecision.UNCERTAIN:
            return SafetyDecision.CONFIRM

        # Low confidence → CONFIRM
        if confidence < self.thresholds.get("deny", 0.5) and decision == SafetyDecision.DENY:
            return SafetyDecision.CONFIRM
        if confidence < self.thresholds.get("allow", 0.999) and decision == SafetyDecision.ALLOW:
            return SafetyDecision.CONFIRM

        return decision

    def _explanation(self, decision: SafetyDecision, effects: list[str], risk_score: float) -> str:
        """Generate human-readable explanation."""
        if decision == SafetyDecision.ALLOW:
            return f"Command appears safe. Effects: {', '.join(effects) if effects else 'none'}"
        elif decision == SafetyDecision.CONFIRM:
            return f"Command requires confirmation. Risk score: {risk_score:.2f}. Effects: {', '.join(effects) if effects else 'none'}"
        elif decision == SafetyDecision.DENY:
            return f"Command blocked. High risk: {risk_score:.2f}. Dangerous effects: {', '.join(effects)}"
        else:
            return f"Uncertain classification. Effects: {', '.join(effects) if effects else 'none'}"


# Integration with existing validator
class SafetyValidator:
    """
    Extended validator that uses SafetyAnalyzer for terminal commands.
    """

    def __init__(self, command_registry, safety_analyzer: SafetyAnalyzer):
        self.registry = command_registry
        self.analyzer = safety_analyzer

    def validate(self, intent) -> tuple[bool, str, bool, str]:
        """
        Validate intent with safety analysis.

        Returns: (valid, error, requires_confirmation, confirmation_message)
        """
        # First, standard validation
        from core.validator import ActionValidator, ValidationResult
        validator = ActionValidator(self.registry)
        result = validator.validate(intent)

        if not result.valid:
            return False, result.error, False, ""

        # If it's a terminal command, run safety analysis
        if intent.tool == "execute_shell":
            tool_request = ToolRequest(
                intent=intent.raw_transcript or "",
                tool=intent.tool,
                shell="bash",
                command=intent.parameters.get("command", ""),
                context={
                    "cwd": "/workspace",
                    "sandbox": "isolated_docker",
                    "network": False,
                    "privileges": "non_root",
                    "scope": "workspace_only",
                },
            )

            safety_result = self.analyzer.analyze(tool_request)

            if safety_result.decision == SafetyDecision.DENY:
                return False, f"Safety check failed: {safety_result.explanation}", False, ""
            elif safety_result.decision == SafetyDecision.CONFIRM:
                return True, "", True, f"Safety check: {safety_result.explanation}"

        return result.valid, "", result.requires_confirmation, result.confirmation_message


# Global instance
_safety_analyzer: Optional[SafetyAnalyzer] = None
_safety_validator: Optional[SafetyValidator] = None


def get_safety_analyzer(model_dir: str = None, parser_shell: str = "bash") -> SafetyAnalyzer:
    """Get or create global safety analyzer."""
    global _safety_analyzer
    if _safety_analyzer is None:
        if model_dir is None:
            # Default to best model from benchmark
            model_dir = "benchmarks/safety_classifier/results/onnx/distilbert_direct"
        _safety_analyzer = SafetyAnalyzer(model_dir, parser_shell)
    return _safety_analyzer


def get_safety_validator(command_registry, model_dir: str = None) -> SafetyValidator:
    """Get or create global safety validator."""
    global _safety_validator
    if _safety_validator is None:
        analyzer = get_safety_analyzer(model_dir)
        _safety_validator = SafetyValidator(command_registry, analyzer)
    return _safety_validator


def close_safety() -> None:
    """Close global safety instances."""
    global _safety_analyzer, _safety_validator
    _safety_analyzer = None
    _safety_validator = None