#!/usr/bin/env python3
"""Train a single candidate model."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SafetyDataset(Dataset):
    """Dataset for safety classification."""

    label2id = {"ALLOW": 0, "CONFIRM": 1, "DENY": 2, "UNCERTAIN": 3}
    id2label = {v: k for k, v in label2id.items()}

    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_file) as f:
            for line in f:
                ex = json.loads(line)
                # Build input text from structured data
                text = self._serialize_example(ex)
                self.examples.append({
                    "text": text,
                    "label": self._label_to_id(ex["label"]),
                    "context": ex.get("context", {}),
                })

    def _label_to_id(self, label: str) -> int:
        return self.label2id.get(label, 0)

    def _serialize_example(self, ex: dict) -> str:
        """Serialize structured example to text."""
        parts = [
            f"intent: {ex.get('intent', '')}",
            f"tool: {ex.get('tool', '')}",
            f"shell: {ex.get('shell', 'bash')}",
            f"command: {ex.get('command', '')}",
        ]

        ctx = ex.get("context", {})
        if ctx:
            ctx_parts = []
            for k, v in ctx.items():
                ctx_parts.append(f"{k}: {v}")
            parts.append(f"context: {', '.join(ctx_parts)}")

        return " [SEP] ".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    # Overall metrics
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    # Per-class metrics
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    class_names = ["ALLOW", "CONFIRM", "DENY", "UNCERTAIN"]

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    # Per-class FNR (for DENY class, FNR = FN / (FN + TP) = cm[2, :].sum() - cm[2,2] / cm[2, :].sum())
    for i, name in enumerate(class_names):
        if cm[i, :].sum() > 0:
            fnr = (cm[i, :].sum() - cm[i, i]) / cm[i, :].sum()
            metrics[f"fnr_{name.lower()}"] = fnr

    return metrics


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
# Load config
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    # Ensure numeric types (YAML may parse as strings)
    training_config["learning_rate"] = float(training_config.get("learning_rate", 2e-4))
    training_config["weight_decay"] = float(training_config.get("weight_decay", 0.01))
    training_config["warmup_ratio"] = float(training_config.get("warmup_ratio", 0.1))
    training_config["max_grad_norm"] = float(training_config.get("max_grad_norm", 1.0))
    training_config["early_stopping_threshold"] = float(training_config.get("early_stopping_threshold", 0.001))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset = SafetyDataset(data_config["train_file"], tokenizer, data_config["max_length"])
    val_dataset = SafetyDataset(data_config["val_file"], tokenizer, data_config["max_length"])

    # Load model
    if model_config["type"] == "effect_extraction":
        # Multi-label
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config["name"],
            num_labels=model_config["num_labels"],
            problem_type="multi_label_classification",
        )
    else:
        # Single-label
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config["name"],
            num_labels=model_config["num_labels"],
            id2label=model_config.get("id2label", {str(i): f"LABEL_{i}" for i in range(model_config["num_labels"])}),
            label2id=model_config.get("label2id", {f"LABEL_{i}": i for i in range(model_config["num_labels"])}),
        )

    # Apply LoRA if enabled
    if model_config.get("lora", {}).get("enabled", False):
        lora_cfg = model_config["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["q_lin", "v_lin"]),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Freeze encoder if specified
    if model_config.get("frozen", False):
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Training arguments
    # Calculate warmup steps from ratio
    train_dataset_size = len(train_dataset)
    steps_per_epoch = train_dataset_size // (training_config.get("batch_size", 16) * training_config.get("gradient_accumulation_steps", 2))
    warmup_steps = int(steps_per_epoch * training_config.get("warmup_ratio", 0.1))
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.get("epochs", 5),
        per_device_train_batch_size=training_config.get("batch_size", 16),
        per_device_eval_batch_size=training_config.get("batch_size", 16),
        learning_rate=training_config.get("learning_rate", 2e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        fp16=training_config.get("fp16", True),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
        eval_strategy="steps",
        eval_steps=training_config.get("eval_steps", 500),
        save_steps=training_config.get("save_steps", 500),
        logging_steps=training_config.get("logging_steps", 100),
        save_total_limit=output_config.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model=training_config.get("metric_for_best_model", "fnr_DENY"),
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=training_config.get("early_stopping_patience", 3),
            early_stopping_threshold=training_config.get("early_stopping_threshold", 0.001),
        )],
    )

    # Train
    trainer.train()

    # Save best model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()