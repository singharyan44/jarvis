#!/usr/bin/env python3
"""Evaluate trained models on test and red-team sets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SafetyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_file) as f:
            for line in f:
                ex = json.loads(line)
                text = self._serialize_example(ex)
                self.examples.append({
                    "text": text,
                    "label": self._label_to_id(ex["label"]),
                    "raw": ex,
                })

        self.label2id = {"ALLOW": 0, "CONFIRM": 1, "DENY": 2, "UNCERTAIN": 3}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def _label_to_id(self, label: str) -> int:
        return {"ALLOW": 0, "CONFIRM": 1, "DENY": 2, "UNCERTAIN": 3}.get(label, 0)

    def _serialize_example(self, ex: dict) -> str:
        parts = [
            f"intent: {ex.get('intent', '')}",
            f"tool: {ex.get('tool', '')}",
            f"shell: {ex.get('shell', 'bash')}",
            f"command: {ex.get('command', '')}",
        ]
        ctx = ex.get("context", {})
        if ctx:
            ctx_parts = [f"{k}: {v}" for k, v in ctx.items()]
            parts.append(f"context: {', '.join(ctx_parts)}")
        return " [SEP] ".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def load_attack_families(config_path: str = "configs/attack_families.yaml") -> dict:
    """Load attack families configuration."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("attack_families", {})


def classify_attack_family(command: str, attack_families: dict) -> list:
    """Classify a command into attack families based on pattern matching."""
    families = []
    cmd_lower = command.lower()
    
    for family_name, family_info in attack_families.items():
        for example in family_info.get("examples", []):
            example_lower = example.lower()
            # Simple substring match for common patterns
            if any(keyword in cmd_lower for keyword in example_lower.split() if len(keyword) > 3):
                families.append(family_name)
                break
    
    return list(set(families))


def compute_family_fnr(labels: np.ndarray, preds: np.ndarray, examples: list, attack_families: dict) -> dict:
    """Compute FNR per attack family."""
    family_fnr = {}
    
    # Get all family names
    family_names = list(attack_families.keys())
    
    for family in family_names:
        family_mask = []
        for ex in examples:
            cmd = ex.get("command", "")
            cmd_families = classify_attack_family(cmd, attack_families)
            family_mask.append(family in cmd_families)
        
        family_mask = np.array(family_mask)
        if family_mask.sum() > 0:
            # For DENY class (label 2), FNR = FN / (FN + TP)
            family_labels = labels[family_mask]
            family_preds = preds[family_mask]
            
            # Count DENY (label 2) predictions vs actual
            deny_mask = (family_labels == 2)
            if deny_mask.sum() > 0:
                fn = np.sum((family_labels == 2) & (family_preds != 2))
                tp = np.sum((family_labels == 2) & (family_preds == 2))
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                family_fnr[family] = float(fnr)
    
    return family_fnr


def compute_technique_fnr(labels: np.ndarray, preds: np.ndarray, examples: list) -> dict:
    """Compute FNR per technique (from redteam metadata)."""
    technique_fnr = {}
    
    # Group by technique
    technique_groups = {}
    for idx, ex in enumerate(examples):
        technique = ex.get("technique", ex.get("mutation_type", "unknown"))
        if technique not in technique_groups:
            technique_groups[technique] = []
        technique_groups[technique].append(idx)
    
    for technique, indices in technique_groups.items():
        if len(indices) > 0:
            tech_labels = labels[indices]
            tech_preds = preds[indices]
            
            # Count DENY (label 2) predictions vs actual
            deny_mask = (tech_labels == 2)
            if deny_mask.sum() > 0:
                fn = np.sum((tech_labels == 2) & (tech_preds != 2))
                tp = np.sum((tech_labels == 2) & (tech_preds == 2))
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                technique_fnr[technique] = float(fnr)
    
    return technique_fnr


def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals for FNR metrics."""
    if len(labels) == 0:
        return {}
    
    # Overall DENY FNR
    deny_mask = (labels == 2)
    if deny_mask.sum() == 0:
        return {}
    
    fnr_values = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(len(labels)), replace=True, random_state=None)
        boot_labels = labels[indices]
        boot_preds = preds[indices]
        
        boot_deny_mask = (boot_labels == 2)
        if boot_deny_mask.sum() > 0:
            fn = np.sum((boot_labels == 2) & (boot_preds != 2))
            tp = np.sum((boot_labels == 2) & (boot_preds == 2))
            if (fn + tp) > 0:
                fnr = fn / (fn + tp)
                fnr_values.append(fnr)
    
    if not fnr_values:
        return {}
    
    fnr_values = np.array(fnr_values)
    alpha = (1 - confidence) / 2
    lower = np.percentile(fnr_values, alpha * 100)
    upper = np.percentile(fnr_values, (1 - alpha) * 100)
    mean_fnr = np.mean(fnr_values)
    
    return {
        "mean": float(mean_fnr),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_samples": int(deny_mask.sum()),
    }


def bootstrap_ci_per_class(labels: np.ndarray, preds: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Compute bootstrap CIs for per-class FNR."""
    class_names = ["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]
    results = {}
    
    for class_idx, name in enumerate(class_names):
        class_mask = (labels == class_idx)
        if class_mask.sum() == 0:
            continue
        
        fnr_values = []
        for _ in range(n_bootstrap):
            indices = resample(np.arange(len(labels)), replace=True, random_state=None)
            boot_labels = labels[indices]
            boot_preds = preds[indices]
            
            boot_class_mask = (boot_labels == class_idx)
            if boot_class_mask.sum() > 0:
                fn = np.sum((boot_labels == class_idx) & (boot_preds != class_idx))
                tp = np.sum((boot_labels == class_idx) & (boot_preds == class_idx))
                if (fn + tp) > 0:
                    fnr = fn / (fn + tp)
                    fnr_values.append(fnr)
        
        if fnr_values:
            fnr_values = np.array(fnr_values)
            alpha = (1 - confidence) / 2
            lower = np.percentile(fnr_values, alpha * 100)
            upper = np.percentile(fnr_values, (1 - alpha) * 100)
            mean_fnr = np.mean(fnr_values)
            
            results[f"fnr_{name.lower()}"] = {
                "mean": float(mean_fnr),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence": confidence,
                "n_bootstrap": n_bootstrap,
                "n_samples": int(class_mask.sum()),
            }
    
    return results


def check_min_samples(labels: np.ndarray, min_samples_per_class: int = 30) -> dict:
    """Check if each class has minimum samples for statistical validity."""
    class_names = ["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]
    results = {}
    
    for class_idx, name in enumerate(class_names):
        count = int(np.sum(labels == class_idx))
        results[name.lower()] = {
            "count": count,
            "meets_minimum": count >= min_samples_per_class,
            "min_required": min_samples_per_class,
        }
    
    return results


def evaluate_model(
    model_path: str,
    test_file: str,
    redteam_file: str,
    config_path: str,
    output_dir: str,
) -> dict:
    """Evaluate a single model."""
    config = load_config(config_path)
    model_name = config["model"]["name"]
    lora_enabled = config["model"].get("lora", {}).get("enabled", False)
    num_labels = config["model"].get("num_labels", 4)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if lora_enabled:
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate on test set
    test_dataset = SafetyDataset(test_file, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    class_names = ["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]

    # Per-class FNR
    fnr_per_class = {}
    for i, name in enumerate(["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]):
        if cm[i, :].sum() > 0:
            fnr = (cm[i, :].sum() - cm[i, i]) / cm[i, :].sum()
            fnr_per_class[f"fnr_{name.lower()}"] = float(fnr)
        else:
            fnr_per_class[f"fnr_{name.lower()}"] = 0.0

    # Calibration metrics (ECE)
    ece = 0.0
    for class_idx in range(4):
        class_mask = (all_labels == class_idx)
        if class_mask.sum() > 0:
            prob_true, prob_pred = calibration_curve(class_mask.astype(int), all_probs[:, class_idx], n_bins=10)
            ece += np.mean(np.abs(prob_true - prob_pred)) * class_mask.sum() / len(all_labels)

    # Brier score
    y_onehot = np.eye(4)[all_labels]
    brier = brier_score_loss(y_onehot.ravel(), all_probs.ravel())

    # Evaluate on red-team set
    redteam_results = {}
    if os.path.exists(redteam_file):
        redteam_dataset = SafetyDataset(redteam_file, tokenizer)
        redteam_loader = torch.utils.data.DataLoader(redteam_dataset, batch_size=32)

        rt_preds = []
        rt_labels = []
        rt_probs = []
        rt_examples = []

        with torch.no_grad():
            for batch in redteam_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                rt_preds.extend(preds.cpu().numpy())
                rt_labels.extend(labels.cpu().numpy())
                rt_probs.extend(probs.cpu().numpy())
                # Get raw examples for technique/family analysis
                for i in range(len(labels)):
                    idx = len(rt_labels) - len(labels) + i
                    if idx < len(redteam_dataset.examples):
                        rt_examples.append(redteam_dataset.examples[idx]["raw"])

        rt_preds = np.array(rt_preds)
        rt_labels = np.array(rt_labels)

        # Red-team specific metrics
        rt_cm = confusion_matrix(rt_labels, rt_preds, labels=[0, 1, 2, 3])
        
        # Per-class FNR
        rt_fnr_per_class = {}
        for i, name in enumerate(["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]):
            if rt_cm[i, :].sum() > 0:
                fnr = (rt_cm[i, :].sum() - rt_cm[i, i]) / rt_cm[i, :].sum()
                rt_fnr_per_class[f"fnr_{name.lower()}"] = float(fnr)
            else:
                rt_fnr_per_class[f"fnr_{name.lower()}"] = 0.0

        # Per-attack-family FNR
        attack_families = load_attack_families()
        family_fnr = compute_family_fnr(rt_labels, rt_preds, rt_examples, attack_families)

        # Per-technique FNR
        technique_fnr = compute_technique_fnr(rt_labels, rt_preds, rt_examples)

        # Bootstrap confidence intervals
        overall_ci = bootstrap_ci(rt_labels, rt_preds)
        per_class_ci = bootstrap_ci_per_class(rt_labels, rt_preds)

        # Minimum sample size check
        min_samples_check = check_min_samples(rt_labels)

        redteam_results = {
            "confusion_matrix": rt_cm.tolist(),
            "fnr_per_class": rt_fnr_per_class,
            "fnr_per_class_ci": per_class_ci,
            "fnr_overall_ci": overall_ci,
            "fnr_by_attack_family": family_fnr,
            "fnr_by_technique": technique_fnr,
            "min_samples_check": min_samples_check,
        }

    # Compile results
    results = {
        "model_path": model_path,
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1_macro),
        "test_f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "fnr_per_class": fnr_per_class,
        "ece": float(ece),
        "brier_score": float(brier),
        "redteam": redteam_results,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--test_file", required=True, help="Test data file")
    parser.add_argument("--redteam_file", required=True, help="Red-team data file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    results = evaluate_model(
        args.model,
        args.test_file,
        args.redteam_file,
        args.config,
        args.output,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {args.output}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Macro: {results['test_f1_macro']:.4f}")
    print(f"FNR per class: {results['fnr_per_class']}")
    if "redteam" in results and results["redteam"]:
        rt = results["redteam"]
        print(f"Red-team FNR (DENY): {rt.get('fnr_per_class', {}).get('fnr_deny', 'N/A')}")
        if "fnr_overall_ci" in rt:
            ci = rt["fnr_overall_ci"]
            print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        if "fnr_by_attack_family" in rt:
            print(f"  By attack family: {rt['fnr_by_attack_family']}")
        if "fnr_by_technique" in rt:
            print(f"  By technique: {rt['fnr_by_technique']}")


if __name__ == "__main__":
    main()