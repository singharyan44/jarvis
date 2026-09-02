#!/usr/bin/env python3
"""Benchmark inference latency for ONNX models."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def load_onnx_model(model_path: str, use_cuda: bool = True) -> ort.InferenceSession:
    """Load ONNX model with optimal settings."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1  # Single thread for latency measurement
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if use_cuda:
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)


def prepare_inputs(tokenizer, texts: list[str], max_length: int = 512) -> list[dict]:
    """Prepare inputs for batch inference."""
    inputs_list = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        inputs_list.append({
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        })
    return inputs_list


def benchmark_model(
    model_path: str,
    tokenizer_path: str,
    test_texts: list[str],
    warmup_runs: int = 100,
    measure_runs: int = 1000,
    batch_sizes: list[int] = [1, 4, 8],
    use_cuda: bool = True,
) -> dict:
    """Benchmark model latency."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    session = load_onnx_model(model_path, use_cuda)

    # Prepare all inputs
    all_inputs = prepare_inputs(tokenizer, test_texts)

    results = {
        "model_path": model_path,
        "batch_results": {},
    }

    for batch_size in batch_sizes:
        print(f"Benchmarking batch_size={batch_size}...")

        # Create batches
        batches = []
        for i in range(0, len(all_inputs), batch_size):
            batch = all_inputs[i:i+batch_size]
            if len(batch) < batch_size:
                # Pad with first input
                while len(batch) < batch_size:
                    batch.append(all_inputs[0])
            batches.append(batch)

        # Warmup
        for _ in range(warmup_runs):
            batch = batches[0]
            input_ids = np.vstack([b["input_ids"] for b in batch])
            attention_mask = np.vstack([b["attention_mask"] for b in batch])
            session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

        # Measure
        latencies = []
        for _ in range(measure_runs):
            for batch in batches:
                input_ids = np.vstack([b["input_ids"] for b in batch])
                attention_mask = np.vstack([b["attention_mask"] for b in batch])

                start = time.perf_counter()
                session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        per_sample = latencies / batch_size

        results["batch_results"][str(batch_size)] = {
            "mean_ms": float(np.mean(per_sample)),
            "median_ms": float(np.median(per_sample)),
            "std_ms": float(np.std(per_sample)),
            "p50_ms": float(np.percentile(per_sample, 50)),
            "p90_ms": float(np.percentile(per_sample, 90)),
            "p95_ms": float(np.percentile(per_sample, 95)),
            "p99_ms": float(np.percentile(per_sample, 99)),
            "p999_ms": float(np.percentile(per_sample, 99.9)),
            "min_ms": float(np.min(per_sample)),
            "max_ms": float(np.max(per_sample)),
            "throughput_per_sec": float(batch_size / (np.mean(latencies) / 1000)),
            "total_runs": len(per_sample),
        }

    # Cold start measurement
    print("Measuring cold start...")
    session = load_onnx_model(model_path)
    cold_latencies = []
    for _ in range(20):
        batch = all_inputs[:1]
        input_ids = np.vstack([b["input_ids"] for b in batch])
        attention_mask = np.vstack([b["attention_mask"] for b in batch])

        start = time.perf_counter()
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        end = time.perf_counter()
        cold_latencies.append((end - start) * 1000)

    results["cold_start"] = {
        "mean_ms": float(np.mean(cold_latencies)),
        "median_ms": float(np.median(cold_latencies)),
        "p95_ms": float(np.percentile(cold_latencies, 95)),
    }

    # Memory usage (approximate)
    results["memory"] = {
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer/model directory path")
    parser.add_argument("--test_file", required=True, help="Test data file for inputs")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--measure", type=int, default=50, help="Measurement runs")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1], help="Batch sizes to test")
    args = parser.parse_args()

    # Load test texts
    test_texts = []
    with open(args.test_file) as f:
        for line in f:
            ex = json.loads(line)
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
            test_texts.append(" [SEP] ".join(parts))
            if len(test_texts) >= 10:  # Limit for speed
                break

    if not test_texts:
        test_texts = [
            "intent: list files [SEP] tool: terminal [SEP] shell: bash [SEP] command: ls [SEP] context: cwd: /workspace",
            "intent: delete all [SEP] tool: terminal [SEP] shell: bash [SEP] command: rm -rf / [SEP] context: cwd: /",
        ]

    print(f"Running latency benchmark on {args.model}")
    results = benchmark_model(
        args.model,
        args.tokenizer,
        test_texts,
        warmup_runs=args.warmup,
        measure_runs=args.measure,
        batch_sizes=args.batch_sizes,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark complete. Results saved to {args.output}")

    # Print summary
    for bs, res in results["batch_results"].items():
        print(f"Batch {bs}: p50={res['p50_ms']:.2f}ms, p95={res['p95_ms']:.2f}ms, p99={res['p99_ms']:.2f}ms, throughput={res['throughput_per_sec']:.1f}/s")

    print(f"Cold start: mean={results['cold_start']['mean_ms']:.2f}ms, p95={results['cold_start']['p95_ms']:.2f}ms")
    print(f"Model size: {results['memory']['model_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()