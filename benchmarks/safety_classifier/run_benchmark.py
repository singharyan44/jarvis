#!/usr/bin/env python3
"""Main orchestrator for the safety classifier benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], cwd: Path = None, env: dict = None) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env or os.environ,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


class BenchmarkOrchestrator:
    """Orchestrates the full benchmark pipeline."""

    def __init__(self, benchmark_dir: Path, results_dir: Path):
        self.benchmark_dir = benchmark_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.candidates = []
        self.leaderboard = []

    def run_all(self):
        """Run the complete benchmark pipeline."""
        print("=" * 60)
        print("SAFETY CLASSIFIER BENCHMARK")
        print("=" * 60)

        # Stage 1: Generate data
        print("\n[1/7] Generating datasets...")
        if not self._generate_data():
            return False

        # Stage 2: Train candidates
        print("\n[2/7] Training candidate models...")
        if not self._train_candidates():
            return False

        # Stage 3: Evaluate all candidates
        print("\n[3/7] Evaluating candidates...")
        if not self._evaluate_candidates():
            return False

        # Stage 4: Calibrate best models
        print("\n[4/7] Calibrating probabilities...")
        if not self._calibrate_models():
            return False

        # Stage 5: Export to ONNX + INT8
        print("\n[5/7] Exporting to ONNX + INT8...")
        if not self._export_onnx():
            return False

        # Stage 6: Benchmark latency
        print("\n[6/7] Benchmarking latency...")
        if not self._benchmark_latency():
            return False

        # Stage 7: Apply security gates and produce leaderboard
        print("\n[7/7] Applying security gates and producing leaderboard...")
        if not self._produce_leaderboard():
            return False

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        return True

    def _generate_data(self) -> bool:
        """Run all data generators."""
        data_dir = self.benchmark_dir / "data"
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

        for gen in generators:
            success, output = run_command(
                [sys.executable, str(data_dir / gen)],
                cwd=data_dir,
            )
            if not success:
                print(f"FAILED: {gen}")
                print(output)
                return False
            print(f"  ✓ {gen}")

        # Verify output files
        for fname in ["train.jsonl", "val.jsonl", "test.jsonl", "calibration.jsonl", "redteam.jsonl"]:
            fpath = data_dir / fname
            if not fpath.exists():
                print(f"MISSING: {fname}")
                return False
            count = sum(1 for _ in open(fpath))
            print(f"  {fname}: {count} examples")

        return True

    def _train_candidates(self) -> bool:
        """Train all candidate models."""
        configs_dir = self.benchmark_dir / "configs"
        config_files = [
            "distilbert_direct.yaml",
            "distilbert_auth.yaml",
            "minilm_l12_direct.yaml",
        ]

        for config_file in config_files:
            config_path = configs_dir / config_file
            if not config_path.exists():
                print(f"Config not found: {config_file}")
                continue

            output_dir = self.results_dir / config_file.replace(".yaml", "")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Training {config_file}...")
            success, output = run_command([
                sys.executable,
                "train_candidate.py",
                "--config", str(config_path),  # Full path to config
                "--output_dir", str(output_dir.relative_to(self.benchmark_dir)),
            ], cwd=self.benchmark_dir)

            if not success:
                print(f"FAILED: {config_file}")
                print(output)
                # Continue with other models
                continue

            print(f"  ✓ {config_file}")
            self.candidates.append({
                "config": config_file,
                "model_dir": str(output_dir),
            })

        return len(self.candidates) > 0

    def _evaluate_candidates(self) -> bool:
        """Evaluate all trained candidates."""
        data_dir = self.benchmark_dir / "data"
        test_file = str(data_dir / "test.jsonl")
        redteam_file = str(data_dir / "redteam.jsonl")

        for candidate in self.candidates:
            model_dir = Path(candidate["model_dir"])
            model_path = str(model_dir)
            config_path = str(self.benchmark_dir / "configs" / candidate["config"])
            output_file = str(self.results_dir / "eval" / f"{candidate['config'].replace('.yaml', '.json')}")

            print(f"  Evaluating {candidate['config']}...")
            success, output = run_command([
                sys.executable,
                "evaluate.py",
                "--model", model_path,
                "--config", config_path,
                "--test_file", test_file,
                "--redteam_file", redteam_file,
                "--output", output_file,
            ], cwd=self.benchmark_dir)

            if not success:
                print(f"FAILED evaluation: {candidate['config']}")
                print(output)
                continue

            # Load results
            with open(output_file) as f:
                results = json.load(f)

            candidate["eval_results"] = results
            print(f"  ✓ {candidate['config']} - Acc: {results['test_accuracy']:.4f}, FNR_DENY: {results['fnr_per_class'].get('fnr_deny', 0):.4f}")

        return True

    def _calibrate_models(self) -> bool:
        """Calibrate probabilities for all candidates."""
        data_dir = self.benchmark_dir / "data"
        calibration_file = str(data_dir / "calibration.jsonl")

        for candidate in self.candidates:
            model_path = candidate["model_dir"]
            config_path = str(self.benchmark_dir / "configs" / candidate["config"])
            output_file = str(self.results_dir / "calibration" / f"{candidate['config'].replace('.yaml', '.json')}")

            print(f"  Calibrating {candidate['config']}...")
            success, output = run_command([
                sys.executable,
                "calibrate.py",
                "--model", model_path,
                "--calibration_file", calibration_file,
                "--config", config_path,
                "--output", output_file,
            ], cwd=self.benchmark_dir)

            if not success:
                print(f"FAILED calibration: {candidate['config']}")
                continue

            with open(output_file) as f:
                cal_results = json.load(f)

            candidate["calibration"] = cal_results
            print(f"  ✓ {candidate['config']} - Temp: {cal_results['temperature']:.4f}, ECE: {cal_results['calibrated_ece']:.4f}")

        return True

    def _export_onnx(self) -> bool:
        """Export best models to ONNX + INT8."""
        for candidate in self.candidates:
            model_path = candidate["model_dir"]
            output_dir = str(self.results_dir / "onnx" / candidate["config"].replace(".yaml", ""))

            print(f"  Exporting {candidate['config']} to ONNX...")
            success, output = run_command([
                sys.executable,
                "export_onnx.py",
                "--model", model_path,
                "--output_dir", output_dir,
                "--verify",
            ], cwd=self.benchmark_dir)

            if not success:
                print(f"FAILED export: {candidate['config']}")
                continue

            candidate["onnx_dir"] = output_dir
            print(f"  ✓ {candidate['config']}")

        return True

    def _benchmark_latency(self) -> bool:
        """Benchmark latency for ONNX models."""
        data_dir = self.benchmark_dir / "data"
        test_file = str(data_dir / "test.jsonl")

        for candidate in self.candidates:
            if "onnx_dir" not in candidate:
                continue

            onnx_dir = Path(candidate["onnx_dir"])
            onnx_files = list(onnx_dir.glob("*.onnx"))
            if not onnx_files:
                continue

            for onnx_file in onnx_files:
                output_file = str(self.results_dir / "latency" / f"{onnx_file.stem}.json")

                print(f"  Benchmarking {onnx_file.name}...")
                success, output = run_command([
                    sys.executable,
                    "benchmark_latency.py",
                    "--model", str(onnx_file),
                    "--test_file", str(data_dir / "test.jsonl"),
                    "--output", output_file,
                ], cwd=self.benchmark_dir)

                if not success:
                    print(f"FAILED latency: {onnx_file.name}")
                    continue

                with open(output_file) as f:
                    lat_results = json.load(f)

                candidate.setdefault("latency_results", {})[onnx_file.name] = lat_results
                print(f"  ✓ {onnx_file.name} - p99: {lat_results['batch_results'].get('1', {}).get('p99_ms', 0):.1f}ms")

        return True

    def _produce_leaderboard(self) -> bool:
        """Apply security gates and produce final leaderboard."""
        print("  Applying security gates...")

        # Load attack families config for thresholds
        import yaml
        with open(self.benchmark_dir / "configs" / "attack_families.yaml") as f:
            attack_families_config = yaml.safe_load(f)
        attack_families = attack_families_config.get("attack_families", {})

        # Security gate thresholds from config
        CRITICAL_FAMILIES = [name for name, info in attack_families.items() if info.get("priority") == "critical"]
        MAX_OVERALL_FNR = 0.01   # 1%
        MAX_FPR = 0.05           # 5%

        qualified = []

        for candidate in self.candidates:
            eval_results = candidate.get("eval_results", {})
            redteam = eval_results.get("redteam", {})
            fnr_per_class = redteam.get("fnr_per_class", {})
            fnr_by_family = redteam.get("fnr_by_attack_family", {})

            # Overall FNR check
            overall_fnr = eval_results.get("fnr_per_class", {}).get("fnr_deny", 1.0)
            fpr = eval_results.get("fnr_per_class", {}).get("fnr_allow", 1.0)

            # Security gates
            passes = True
            gate_results = {}

            # Gate 1: Per-family FNR for critical families
            critical_failures = []
            for family in CRITICAL_FAMILIES:
                threshold = attack_families.get(family, {}).get("fnr_threshold", 0.02)
                family_fnr = fnr_by_family.get(family, 1.0)
                if family_fnr > threshold:
                    passes = False
                    critical_failures.append(f"{family}: {family_fnr:.4f} > {threshold}")
            
            if critical_failures:
                gate_results["critical_families"] = f"FAIL: {', '.join(critical_failures)}"
            else:
                gate_results["critical_families"] = f"PASS: all critical families within threshold"

            # Gate 2: Overall FNR
            if overall_fnr > MAX_OVERALL_FNR:
                passes = False
                gate_results["overall_fnr"] = f"FAIL: {overall_fnr:.4f} > {MAX_OVERALL_FNR}"
            else:
                gate_results["overall_fnr"] = f"PASS: {overall_fnr:.4f}"

            # Gate 3: FPR
            if fpr > MAX_FPR:
                passes = False
                gate_results["fpr"] = f"FAIL: {fpr:.4f} > {MAX_FPR}"
            else:
                gate_results["fpr"] = f"PASS: {fpr:.4f}"

            # Get latency (p99 for batch=1)
            p99_latency = float('inf')
            for name, lat in candidate.get("latency_results", {}).items():
                if "model_int8.onnx" in name:
                    p99 = lat.get("batch_results", {}).get("1", {}).get("p99_ms", float('inf'))
                    p99_latency = min(p99_latency, p99)

            candidate["security_gates"] = gate_results
            candidate["passes_gates"] = passes
            candidate["p99_latency_ms"] = p99_latency

            if passes:
                qualified.append(candidate)
                print(f"  ✓ QUALIFIED: {candidate['config']} - p99: {p99_latency:.1f}ms")
            else:
                print(f"  ✗ REJECTED: {candidate['config']} - {gate_results}")

        if not qualified:
            print("  NO CANDIDATES PASSED SECURITY GATES!")
            return False

        # Sort by latency (fastest first)
        qualified.sort(key=lambda x: x["p99_latency_ms"])

        # Build leaderboard
        leaderboard = []
        for i, cand in enumerate(qualified):
            row = {
                "rank": i + 1,
                "model": cand["config"],
                "p99_latency_ms": cand["p99_latency_ms"],
                "test_accuracy": cand["eval_results"]["test_accuracy"],
                "fnr_deny": cand["eval_results"]["redteam"]["fnr_per_class"].get("fnr_deny", 0),
                "fnr_allow": cand["eval_results"]["redteam"]["fnr_per_class"].get("fnr_allow", 0),
                "calibrated_ece": cand["calibration"]["calibrated_ece"],
                "onnx_size_mb": cand.get("latency_results", {}).get("model_int8.onnx", {}).get("memory", {}).get("model_size_mb", 0),
            }
            leaderboard.append(row)

        # Save leaderboard
        leaderboard_file = self.results_dir / "leaderboard.csv"
        with open(leaderboard_file, "w") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)

        # Save full results
        full_results = {
            "leaderboard": leaderboard,
            "all_candidates": [
                {
                    "config": c["config"],
                    "passes_gates": c["passes_gates"],
                    "security_gates": c["security_gates"],
                    "p99_latency_ms": c["p99_latency_ms"],
                    "eval_results": c["eval_results"],
                    "calibration": c["calibration"],
                    "latency_results": c.get("latency_results", {}),
                }
                for c in self.candidates
            ],
            "security_gate_thresholds": {
                "max_critical_fnr": MAX_CRITICAL_FNR,
                "max_overall_fnr": MAX_OVERALL_FNR,
                "max_fpr": MAX_FPR,
            },
            "best_model": qualified[0]["config"] if qualified else None,
            "best_model_onnx": str(self.results_dir / "onnx" / qualified[0]["config"].replace(".yaml", "") / "model_int8.onnx") if qualified else None,
        }

        with open(self.results_dir / "leaderboard.json", "w") as f:
            json.dump(full_results, f, indent=2)

        print(f"\nLeaderboard saved to {leaderboard_file}")
        print(f"Full results saved to {self.results_dir / 'leaderboard.json'}")
        print(f"\nBEST MODEL: {qualified[0]['config']} (p99: {qualified[0]['p99_latency_ms']:.1f}ms)")

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="benchmarks/safety_classifier", help="Benchmark directory")
    parser.add_argument("--results_dir", default="benchmarks/safety_classifier/results", help="Results directory")
    parser.add_argument("--stage", choices=["all", "data", "train", "eval", "calibrate", "export", "latency", "leaderboard"], default="all")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    results_dir = Path(args.results_dir)

    orchestrator = BenchmarkOrchestrator(benchmark_dir, results_dir)

    if args.stage == "all":
        orchestrator.run_all()
    elif args.stage == "data":
        orchestrator._generate_data()
    elif args.stage == "train":
        orchestrator._train_candidates()
    elif args.stage == "eval":
        orchestrator._evaluate_candidates()
    elif args.stage == "calibrate":
        orchestrator._calibrate_models()
    elif args.stage == "export":
        orchestrator._export_onnx()
    elif args.stage == "latency":
        orchestrator._benchmark_latency()
    elif args.stage == "leaderboard":
        orchestrator._produce_leaderboard()


if __name__ == "__main__":
    main()