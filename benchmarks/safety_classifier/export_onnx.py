#!/usr/bin/env python3
"""Export trained model to ONNX and INT8 quantization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import yaml


def export_onnx(
    model_path: str,
    output_path: str,
    opset: int = 17,
    config_path: str = None,
) -> bool:
    """Export PyTorch model to ONNX."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config_path:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_name = config["model"]["name"]
        lora_enabled = config["model"].get("lora", {}).get("enabled", False)
        num_labels = config["model"].get("num_labels", 4)
    else:
        model_name = model_path
        lora_enabled = False
        num_labels = 4

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if lora_enabled:
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    # Dummy input
    dummy_input = tokenizer(
        "intent: test [SEP] tool: terminal [SEP] shell: bash [SEP] command: ls [SEP] context: cwd: /workspace",
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    }

    try:
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        print(f"Exported ONNX model to {output_path}")
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def quantize_int8(onnx_path: str, output_path: str) -> bool:
    """Quantize ONNX model to INT8 using ONNX Runtime."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized INT8 model saved to {output_path}")
        return True
    except ImportError:
        print("onnxruntime not available for quantization")
        return False
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False


def verify_onnx(
    pytorch_model_path: str,
    onnx_path: str,
    test_inputs: list[str],
    tolerance: float = 1e-3,
    config_path: str = None,
) -> dict:
    """Verify ONNX model matches PyTorch outputs."""
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with config
    if config_path:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_name = config["model"]["name"]
        lora_enabled = config["model"].get("lora", {}).get("enabled", False)
        num_labels = config["model"].get("num_labels", 4)
    else:
        model_name = pytorch_model_path
        lora_enabled = False
        num_labels = 4

    # Load PyTorch model
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if lora_enabled:
        pt_model = PeftModel.from_pretrained(pt_model, pytorch_model_path)
    pt_model.eval()

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    results = {
        "matches": 0,
        "total": len(test_inputs),
        "max_logit_diff": 0.0,
        "max_prob_diff": 0.0,
        "details": [],
    }

    for text in test_inputs:
        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=512, truncation=True
        )
        with torch.no_grad():
            pt_outputs = pt_model(**inputs)
            pt_logits = pt_outputs.logits
            pt_probs = torch.softmax(pt_logits, dim=-1).numpy()

        # ONNX
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        ort_logits = ort_session.run(None, ort_inputs)[0]
        ort_probs = np.exp(ort_logits) / np.sum(np.exp(ort_logits), axis=-1, keepdims=True)

        # Compare
        logit_diff = np.max(np.abs(pt_logits.numpy() - ort_logits))
        prob_diff = np.max(np.abs(pt_probs - ort_probs))

        match = logit_diff < tolerance and prob_diff < tolerance
        results["matches"] += int(match)
        results["max_logit_diff"] = max(results["max_logit_diff"], float(logit_diff))
        results["max_prob_diff"] = max(results["max_prob_diff"], float(prob_diff))
        results["details"].append({
            "input": text[:100],
            "logit_diff": float(logit_diff),
            "prob_diff": float(prob_diff),
            "match": match,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PyTorch model path")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX matches PyTorch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    # Export ONNX
    if not export_onnx(args.model, str(onnx_path), args.opset, args.config):
        return

    # Quantize INT8
    if not quantize_int8(str(onnx_path), str(int8_path)):
        print("INT8 quantization skipped")

    # Verify if requested
    if args.verify:
        test_inputs = [
            "intent: list files [SEP] tool: terminal [SEP] shell: bash [SEP] command: ls [SEP] context: cwd: /workspace",
            "intent: delete all [SEP] tool: terminal [SEP] shell: bash [SEP] command: rm -rf / [SEP] context: cwd: /",
            "intent: read key [SEP] tool: terminal [SEP] shell: bash [SEP] command: cat ~/.ssh/id_rsa [SEP] context: cwd: /home/user",
        ]

        results = verify_onnx(args.model, str(onnx_path), test_inputs, config_path=args.config)
        print(f"Verification: {results['matches']}/{results['total']} matched")
        print(f"Max logit diff: {results['max_logit_diff']:.6f}")
        print(f"Max prob diff: {results['max_prob_diff']:.6f}")

        with open(Path(args.output_dir) / "verification.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        print("ONNX export complete. Skipping verification (use --verify to enable)")

    print(f"Export complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()