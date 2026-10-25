#!/usr/bin/env python3
"""CLI entry point for Milestone 5 model evaluation.

Evaluates baseline teachers, M3 self-learning teachers, and M4 distilled/hard-label students
under consistent metric definitions, CPU latency benchmarking, and strict split isolation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.data import PositionRecord
from chessevolve.dataset import generate_random_positions, split_records, build_examples
from chessevolve.distillation import load_records, validate_records_against_manifest
from chessevolve.engine import EngineSettings, StockfishAdapter
from chessevolve.evaluation import (
    evaluate_model_suite,
    save_evaluation_artifacts,
)


DEFAULT_MODELS = [
    {
        "name": "Baseline Teacher",
        "role": "baseline_teacher",
        "checkpoint_path": "experiments/results/medium/best.pt",
        "notes": "Original baseline MLP teacher from M2",
    },
    {
        "name": "M3 Retained Teacher",
        "role": "retained_teacher",
        "checkpoint_path": "experiments/results/sl_round1/retained.pt",
        "notes": "Selected teacher from M3 (baseline retained after candidate rejection)",
    },
    {
        "name": "M3 Rejected Candidate",
        "role": "rejected_candidate_audit",
        "checkpoint_path": "experiments/results/sl_round1/candidate.pt",
        "notes": "Candidate teacher rejected by M3 self-learning gate (regret degraded)",
    },
    {
        "name": "Hard Student (Seed 0)",
        "role": "student_hard",
        "checkpoint_path": "experiments/results/m4_smoke/hard_seed0.pt",
        "notes": "Hard-label student trained on reference best move (seed 0)",
    },
    {
        "name": "Distilled Student (Seed 0)",
        "role": "student_distilled",
        "checkpoint_path": "experiments/results/m4_smoke/distilled_seed0.pt",
        "notes": "Distilled student trained on soft teacher targets (seed 0)",
    },
    {
        "name": "Hard Student (Seed 1)",
        "role": "student_hard",
        "checkpoint_path": "experiments/results/m4_smoke/hard_seed1.pt",
        "notes": "Hard-label student trained on reference best move (seed 1)",
    },
    {
        "name": "Distilled Student (Seed 1)",
        "role": "student_distilled",
        "checkpoint_path": "experiments/results/m4_smoke/distilled_seed1.pt",
        "notes": "Distilled student trained on soft teacher targets (seed 1)",
    },
]


def ensure_test_records_exist(
    manifest_path: Path,
    target_path: Path,
    stockfish_path: Path,
    engine_depth: int = 2,
    seed: int = 7,
) -> Path:
    """Generate and evaluate final-test records matching manifest if not already on disk."""
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_games = set(manifest["source_game_ids"].get("test", []))

    # Generate the positions using the declared seed
    all_records = generate_random_positions(num_games=4, max_plies=8, seed=seed)
    sp = split_records(all_records, seed=seed)
    test_recs = [r for r in sp["test"] if r.source_game_id in test_games]

    settings = EngineSettings(depth=engine_depth)
    with StockfishAdapter(stockfish_path) as adapter:
        adapter.reset()
        examples, skipped = build_examples(test_recs, adapter, settings)

    if skipped:
        raise RuntimeError(f"Failed evaluating test records with Stockfish: {skipped}")

    with open(target_path, "w", encoding="utf-8") as f:
        for ex in examples:
            if ex.record:
                f.write(json.dumps(dataclasses.asdict(ex.record)) + "\n")

    return target_path


def format_results_table(summary: dict) -> str:
    """Format markdown table of evaluated models."""
    split = summary.get("split_name", "").upper()
    is_final = summary.get("is_final_test", False)
    header = f"### Model Evaluation Table ({split} Set {'[FINAL TEST]' if is_final else '[DEVELOPMENT]'})"

    lines = [
        header,
        "",
        "| Model | Role | Top-1 Agreement | Mean Regret (cp) | Loss | Positions | Parameters | Size (KiB) | CPU Latency (ms/pos) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for m in summary.get("models", []):
        top1_pct = f"{m['top1_agreement'] * 100.0:.1f}%"
        regret = f"{m['mean_regret_cp']:.1f}"
        loss = f"{m['loss']:.4f}"
        params = f"{m['trainable_parameters']:,}"
        size = f"{m['checkpoint_size_kib']:.1f}"
        latency = f"{m['latency_ms_per_pos_median']:.3f} ± {m['latency_ms_per_pos_std']:.3f}"
        lines.append(
            f"| {m['model_name']} | `{m['role']}` | {top1_pct} | {regret} | {loss} | {m['num_evaluated_positions']} | {params} | {size} | {latency} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChessEvolve model checkpoints on development or test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Evaluation split to evaluate on (strictly isolated).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset JSONL records file. Defaults to m4_data/datasets/<split>_records.jsonl.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/results/m4_data/datasets/manifest.json"),
        help="Path to frozen split manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results and charts. Defaults to experiments/results/m5_<split>.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations for inference latency measurement.",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=50,
        help="Number of repeated timed runs for inference latency measurement.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=Path,
        default=Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal"),
        help="Path to Stockfish engine executable (used if test records need generation).",
    )

    args = parser.parse_args()

    # Determine default paths
    split = args.split
    output_dir = args.output_dir or Path(f"experiments/results/m5_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    dataset_path = args.dataset
    if dataset_path is None:
        dataset_dir = manifest_path.parent
        if split == "dev":
            dataset_path = dataset_dir / "dev_records.jsonl"
        else:
            dataset_path = dataset_dir / "test_records.jsonl"
            ensure_test_records_exist(
                manifest_path=manifest_path,
                target_path=dataset_path,
                stockfish_path=args.stockfish_path,
                engine_depth=2,
                seed=7,
            )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading and validating {split} records from {dataset_path}...")
    records = validate_records_against_manifest(
        load_records(dataset_path), manifest_path, split
    )
    print(f"Validated {len(records)} {split} records successfully against {manifest_path.name}.")

    # Filter models to those whose checkpoints exist
    available_models = []
    missing_models = []
    for mcfg in DEFAULT_MODELS:
        ckpt_path = Path(mcfg["checkpoint_path"])
        if ckpt_path.is_file():
            available_models.append(mcfg)
        else:
            missing_models.append(mcfg)

    if missing_models:
        print(f"Warning: {len(missing_models)} checkpoints were not found:")
        for m in missing_models:
            print(f"  - {m['name']}: {m['checkpoint_path']}")

    if not available_models:
        raise RuntimeError("No available model checkpoints found to evaluate!")

    print(f"Evaluating {len(available_models)} checkpoints on {len(records)} {split} positions...")
    summary = evaluate_model_suite(
        models_config=available_models,
        records=records,
        manifest_path=manifest_path,
        split_name=split,
        warmup_runs=args.latency_warmup,
        timed_runs=args.latency_runs,
    )

    # Save artifacts
    json_path, csv_path, charts = save_evaluation_artifacts(summary, output_dir)
    print(f"\nArtifacts successfully written to {output_dir}:")
    print(f"  - JSON report: {json_path}")
    print(f"  - CSV table:   {csv_path}")
    for c in charts:
        print(f"  - Chart:       {c}")

    # Output formatted table
    print("\n" + format_results_table(summary) + "\n")


if __name__ == "__main__":
    main()
