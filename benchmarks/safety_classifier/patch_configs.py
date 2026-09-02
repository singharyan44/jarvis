#!/usr/bin/env python3
"""Patch all config YAMLs for RTX 3050 (4GB VRAM) and increase calibration set."""

import yaml
from pathlib import Path

configs_dir = Path("benchmarks/safety_classifier/configs")

# RTX 3050 (4GB) optimized settings
RTX_3050_CONFIG = {
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,  # effective batch = 16
        "fp16": True,
    }
}

# Configs to patch
config_files = [
    "distilbert_direct.yaml",
    "distilbert_effect.yaml",
    "distilbert_auth.yaml",
    "minilm_l6_effect.yaml",
    "minilm_l12_direct.yaml",
    "deberta_direct.yaml",
    "baselines.yaml",
]

for fname in config_files:
    fpath = Path("benchmarks/safety_classifier/configs") / fname
    if not fpath.exists():
        print(f"Skipping missing: {fname}")
        continue

    with open(fpath) as f:
        config = yaml.safe_load(f)

    # Apply RTX 3050 settings
    if "training" in config:
        config["training"].update(RTX_3050_CONFIG["training"])
    else:
        config["training"] = RTX_3050_CONFIG["training"]

    with open(fpath, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Patched: {fname}")

print("\nAll configs patched for RTX 3050 (4GB VRAM)")
print("  batch_size: 16 -> 8")
print("  gradient_accumulation_steps: 2 (effective batch = 16)")
print("  fp16: true")