#!/usr/bin/env python3
"""Run all data generators."""

import subprocess
import sys
from pathlib import Path


def run_generator(script_name: str) -> bool:
    """Run a generator script."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print(f"ERROR running {script_name}:")
        print(result.stderr)
        return False

    print(result.stdout)
    return True


def main():
    generators = [
        "generate_train.py",
        "generate_val.py",
        "generate_test.py",
        "generate_calibration.py",
        "generate_redteam_stage1.py",
        "generate_redteam_stage2.py",
        "generate_redteam_stage3.py",
        "combine_redteam.py",
    ]

    print("Running all data generators...")
    failed = []

    for gen in generators:
        if not run_generator(gen):
            failed.append(gen)

    if failed:
        print(f"\nFAILED generators: {failed}")
        sys.exit(1)

    print("\nAll generators completed successfully!")

    # Verify output files
    data_dir = Path(__file__).parent
    for fname in ["train.jsonl", "val.jsonl", "test.jsonl", "calibration.jsonl", "redteam.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            count = sum(1 for _ in open(fpath))
            print(f"  {fname}: {count} examples")
        else:
            print(f"  {fname}: MISSING")


if __name__ == "__main__":
    main()