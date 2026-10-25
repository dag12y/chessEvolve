"""Model evaluation pipeline, metrics, latency benchmarking, and artifact generation (Milestone 5).

Provides consistent evaluation of baseline teachers, evolved/retained teachers,
and distilled/hard-label student checkpoints on isolated development and final-test sets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .data import PositionRecord
from .dataset import TrainingExample
from .distillation import _record_to_example, load_records, validate_records_against_manifest
from .model import MLPRanker
from .train import evaluate as baseline_evaluate, load_checkpoint


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_model_metadata(checkpoint_path: Path) -> dict:
    """Extract model parameters count, checkpoint size, and hash without modifying checkpoint."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    size_bytes = os.path.getsize(p)
    sha256 = compute_file_sha256(p)
    model, ckpt_dict = load_checkpoint(p)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    return {
        "checkpoint_path": str(p),
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_size_kib": round(size_bytes / 1024, 2),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "model_config": ckpt_dict.get("model_config", {}),
    }


def records_to_evaluation_examples(
    records: Iterable[PositionRecord],
    expected_split: Optional[str] = None,
) -> List[TrainingExample]:
    """Convert valid PositionRecord objects into TrainingExamples for evaluation.

    Safe against empty records. Rejects records with mismatched splits if expected_split is specified.
    """
    examples = []
    for rec in records:
        if expected_split is not None and rec.split != expected_split:
            raise ValueError(
                f"Record {rec.position_id} split {rec.split!r} does not match "
                f"expected split {expected_split!r}"
            )
        examples.append(_record_to_example(rec))
    return examples


def evaluate_model_performance(
    model: MLPRanker,
    examples: List[TrainingExample],
) -> dict:
    """Evaluate model top-1 reference agreement, mean regret (cp), and loss.

    Strictly reuses the established M1/M2 evaluation metric definitions.
    Safely handles empty datasets and non-finite outputs.
    """
    if not examples:
        return {
            "loss": 0.0,
            "top1_agreement": 0.0,
            "mean_regret_cp": 0.0,
            "num_examples": 0,
        }

    # Verify model outputs and candidate count consistency
    model.eval()
    with torch.no_grad():
        for ex in examples:
            pos = torch.from_numpy(ex.position_features).float()
            moves = torch.from_numpy(ex.move_features).float()
            logits = model(pos, moves)
            if logits.shape[0] != len(ex.candidate_scores):
                raise ValueError(
                    f"Model logits count ({logits.shape[0]}) does not match "
                    f"candidate move count ({len(ex.candidate_scores)}) for {ex.position_id}"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits produced for position {ex.position_id}")

    # Use existing M1/M2 evaluate function
    metrics = baseline_evaluate(model, examples)

    # Ensure all return values are strictly finite
    for k in ["loss", "top1_agreement", "mean_regret_cp"]:
        val = metrics.get(k, 0.0)
        if not np.isfinite(val):
            raise FloatingPointError(f"Metric {k} evaluated to non-finite value: {val}")

    return metrics


def measure_inference_latency(
    model: MLPRanker,
    examples: List[TrainingExample],
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> dict:
    """Measure CPU inference latency with warm-up and repeated measurements.

    Separates neural network inference time from any engine or preprocessing latency.
    Reports median, mean, and standard deviation in milliseconds per position.
    """
    if not examples:
        return {
            "latency_ms_per_pos_median": 0.0,
            "latency_ms_per_pos_mean": 0.0,
            "latency_ms_per_pos_std": 0.0,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
        }

    model.eval()
    prepped = [
        (torch.from_numpy(ex.position_features).float(), torch.from_numpy(ex.move_features).float())
        for ex in examples
    ]

    with torch.no_grad():
        # Warmup phase
        for _ in range(max(1, warmup_runs)):
            for pos, moves in prepped:
                _ = model(pos, moves)

        # Timed repetitions
        latencies_ms = []
        for _ in range(max(1, timed_runs)):
            t0 = time.perf_counter()
            for pos, moves in prepped:
                _ = model(pos, moves)
            t1 = time.perf_counter()
            # Latency per individual position in milliseconds
            per_pos_ms = ((t1 - t0) / len(prepped)) * 1000.0
            latencies_ms.append(per_pos_ms)

    return {
        "latency_ms_per_pos_median": float(np.median(latencies_ms)),
        "latency_ms_per_pos_mean": float(np.mean(latencies_ms)),
        "latency_ms_per_pos_std": float(np.std(latencies_ms)),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    examples: List[TrainingExample],
    model_name: str,
    role: str,
    split_name: str,
    warmup_runs: int = 10,
    timed_runs: int = 50,
    notes: Optional[str] = None,
) -> dict:
    """Evaluate a single model checkpoint for metrics, metadata, and latency."""
    p = Path(checkpoint_path)
    metadata = get_model_metadata(p)
    model, _ = load_checkpoint(p)

    perf_metrics = evaluate_model_performance(model, examples)
    latency_metrics = measure_inference_latency(
        model, examples, warmup_runs=warmup_runs, timed_runs=timed_runs
    )

    return {
        "model_name": model_name,
        "role": role,
        "split": split_name,
        "checkpoint_path": str(p),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_size_bytes": metadata["checkpoint_size_bytes"],
        "checkpoint_size_kib": metadata["checkpoint_size_kib"],
        "trainable_parameters": metadata["trainable_parameters"],
        "total_parameters": metadata["total_parameters"],
        "num_evaluated_positions": perf_metrics["num_examples"],
        "top1_agreement": perf_metrics["top1_agreement"],
        "mean_regret_cp": perf_metrics["mean_regret_cp"],
        "loss": perf_metrics["loss"],
        "latency_ms_per_pos_median": latency_metrics["latency_ms_per_pos_median"],
        "latency_ms_per_pos_mean": latency_metrics["latency_ms_per_pos_mean"],
        "latency_ms_per_pos_std": latency_metrics["latency_ms_per_pos_std"],
        "latency_warmup_runs": latency_metrics["warmup_runs"],
        "latency_timed_runs": latency_metrics["timed_runs"],
        "notes": notes or "",
    }


def evaluate_model_suite(
    models_config: List[dict],
    records: List[PositionRecord],
    manifest_path: Optional[Path] = None,
    split_name: str = "dev",
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> dict:
    """Evaluate a full suite of models on the identical evaluation set.

    Validates records against the split manifest to ensure strict isolation.
    """
    if manifest_path is not None:
        p_manifest = Path(manifest_path)
        if not p_manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        validated_records = validate_records_against_manifest(records, p_manifest, split_name)
        manifest_sha256 = compute_file_sha256(p_manifest)
    else:
        validated_records = list(records)
        manifest_sha256 = None

    examples = records_to_evaluation_examples(validated_records, expected_split=split_name)

    evaluated_models = []
    for mcfg in models_config:
        ckpt_path = Path(mcfg["checkpoint_path"])
        res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            examples=examples,
            model_name=mcfg["name"],
            role=mcfg["role"],
            split_name=split_name,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            notes=mcfg.get("notes"),
        )
        evaluated_models.append(res)

    suite_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_name": split_name,
        "is_final_test": split_name == "test",
        "num_positions": len(examples),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": manifest_sha256,
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "models": evaluated_models,
    }
    return suite_summary


def generate_evaluation_charts(
    results: List[dict],
    output_dir: Path,
    split_name: str,
) -> List[Path]:
    """Generate publication-ready charts from actual evaluated results.

    - Regret bar chart (centipawns, lower is better)
    - Top-1 agreement bar chart (%, higher is better)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = [m["model_name"] for m in results]
    regrets = [m["mean_regret_cp"] for m in results]
    agreements = [m["top1_agreement"] * 100.0 for m in results]

    generated_charts = []

    # Chart 1: Mean Engine-Score Regret (Lower is better)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, regrets, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean Engine-Score Regret (Centipawns)", fontsize=12)
    ax.set_title(f"Model Regret Comparison ({split_name.upper()} Set) — Lower is Better", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f} cp",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    regret_chart_path = output_dir / f"regret_by_model_{split_name}.png"
    plt.savefig(regret_chart_path, dpi=200)
    plt.close(fig)
    generated_charts.append(regret_chart_path)

    # Chart 2: Top-1 Agreement (Higher is better)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, agreements, color="#2e7d32", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Top-1 Stockfish Agreement (%)", fontsize=12)
    ax.set_title(f"Top-1 Agreement with Stockfish ({split_name.upper()} Set) — Higher is Better", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(max(agreements) * 1.25, 50))
    plt.xticks(rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    agreement_chart_path = output_dir / f"top1_agreement_by_model_{split_name}.png"
    plt.savefig(agreement_chart_path, dpi=200)
    plt.close(fig)
    generated_charts.append(agreement_chart_path)

    return generated_charts


def save_evaluation_artifacts(
    eval_summary: dict,
    output_dir: Path,
) -> Tuple[Path, Path, List[Path]]:
    """Save machine-readable JSON and CSV tables as well as visual charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_name = eval_summary.get("split_name", "dev")

    # 1. JSON report
    json_path = out / f"evaluation_results_{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # 2. CSV report
    csv_path = out / f"evaluation_results_{split_name}.csv"
    models = eval_summary.get("models", [])
    if models:
        fields = [
            "model_name",
            "role",
            "split",
            "top1_agreement",
            "mean_regret_cp",
            "loss",
            "num_evaluated_positions",
            "trainable_parameters",
            "checkpoint_size_bytes",
            "checkpoint_size_kib",
            "latency_ms_per_pos_median",
            "latency_ms_per_pos_mean",
            "latency_ms_per_pos_std",
            "checkpoint_path",
            "checkpoint_sha256",
            "notes",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for m in models:
                writer.writerow(m)

    # 3. Charts
    charts = generate_evaluation_charts(models, out, split_name)

    return json_path, csv_path, charts
