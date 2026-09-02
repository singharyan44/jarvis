#!/usr/bin/env python3
"""Calibrate model probabilities using temperature scaling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
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
                })

        self.label2id = {"ALLOW": 0, "CONFIRM": 1, "DENY": 2, "UNCERTAIN": 3}

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


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return logits / temperature


def nll_loss(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood loss for temperature scaling."""
    scaled = logits / temperature
    probs = np.exp(scaled) / np.sum(np.exp(scaled), axis=1, keepdims=True)
    nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12)
    return np.mean(nll)


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal temperature using scalar optimization."""
    result = minimize_scalar(
        lambda t: nll_loss(t, logits, labels),
        bounds=(0.01, 10.0),
        method='bounded'
    )
    return result.x


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.mean(in_bin)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--calibration_file", required=True, help="Calibration data file")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--output", required=True, help="Output calibration JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["name"]
    lora_enabled = config["model"].get("lora", {}).get("enabled", False)
    num_labels = config["model"].get("num_labels", 4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if lora_enabled:
        model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load calibration dataset
    cal_dataset = SafetyDataset(args.calibration_file, tokenizer)
    cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=32)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in cal_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)

    # Get uncalibrated probs
    uncal_probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
    uncal_ece = compute_ece(uncal_probs, all_labels)
    uncal_nll = -np.mean(np.log(uncal_probs[np.arange(len(all_labels)), all_labels] + 1e-12))

    # Find optimal temperature
    optimal_temp = find_optimal_temperature(all_logits, all_labels)

    # Calibrated probs
    cal_probs = np.exp(all_logits / optimal_temp) / np.sum(np.exp(all_logits / optimal_temp), axis=1, keepdims=True)
    cal_ece = compute_ece(cal_probs, all_labels)
    cal_nll = -np.mean(np.log(cal_probs[np.arange(len(all_labels)), all_labels] + 1e-12))

    # Save calibration results
    results = {
        "temperature": float(optimal_temp),
        "uncalibrated_ece": float(uncal_ece),
        "uncalibrated_nll": float(uncal_nll),
        "calibrated_ece": float(cal_ece),
        "calibrated_nll": float(cal_nll),
        "num_samples": len(all_labels),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Optimal temperature: {optimal_temp:.4f}")
    print(f"Uncalibrated ECE: {uncal_ece:.4f} -> Calibrated ECE: {cal_ece:.4f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()