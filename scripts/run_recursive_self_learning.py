#!/usr/bin/env python3
"""Run bounded recursive self-learning on a frozen labeled dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.distillation import (  # noqa: E402
    _record_to_example,
    load_records,
    validate_records_against_manifest,
)
from chessevolve.evaluation import evaluate_model_suite  # noqa: E402
from chessevolve.self_learning import (  # noqa: E402
    GateConfig,
    RecursiveSelfLearningConfig,
    SelfLearningRoundConfig,
    VerifierConfig,
    run_recursive_self_learning,
    evaluate_regret_metrics,
)
from chessevolve.train import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run verified recursive self-learning without using final-test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--num-candidate-positions", type=int, default=8)
    parser.add_argument("--selection-policy", choices=["random", "uncertainty", "disagreement"], default="uncertainty")
    parser.add_argument("--engine-depth", type=int, default=4)
    parser.add_argument("--stockfish-path", type=Path, default=Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal"))
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-lr", type=float, default=5e-4)
    parser.add_argument("--fine-tune-batch-size", type=int, default=4)
    parser.add_argument("--min-verified-examples", type=int, default=2)
    parser.add_argument("--gate-min-regret-reduction", type=float, default=0.0)
    parser.add_argument("--gate-max-agreement-drop", type=float, default=0.05)
    parser.add_argument("--min-score-margin", type=int, default=None)
    parser.add_argument("--feature-group", choices=("A", "B", "C", "D"), default="D")
    args = parser.parse_args()

    seeds = args.seed or list(range(len(args.baseline_checkpoint)))
    if len(args.baseline_checkpoint) not in {1, len(seeds)}:
        parser.error("provide one baseline checkpoint or one checkpoint per seed")

    manifest = args.manifest.resolve()
    dataset_dir = args.dataset_dir.resolve()
    train_records = validate_records_against_manifest(
        load_records(dataset_dir / "train_records.jsonl"), manifest, "train"
    )
    dev_records = validate_records_against_manifest(
        load_records(dataset_dir / "dev_records.jsonl"), manifest, "dev"
    )
    test_records = validate_records_against_manifest(
        load_records(dataset_dir / "test_records.jsonl"), manifest, "test"
    )
    valid_test_records = []
    invalid_test_ids = []
    for record in test_records:
        try:
            _record_to_example(record)
        except ValueError as exc:
            if "board is not a valid position" not in str(exc):
                raise
            invalid_test_ids.append(record.position_id)
            continue
        valid_test_records.append(record)
    labeling_summary = json.loads((dataset_dir / "labeling_summary.json").read_text(encoding="utf-8"))
    labeling_depth = labeling_summary["engine_settings"]["depth"]
    if labeling_depth != args.engine_depth:
        parser.error(
            f"dataset labeling depth is {labeling_depth}, but --engine-depth is {args.engine_depth}"
        )
    for record in train_records + dev_records + test_records:
        if record.engine_settings is None or record.engine_settings.get("depth") != args.engine_depth:
            parser.error(
                f"record {record.position_id} has engine depth "
                f"{None if record.engine_settings is None else record.engine_settings.get('depth')}; "
                f"expected {args.engine_depth}"
            )

    gate = GateConfig(
        min_regret_reduction_cp=args.gate_min_regret_reduction,
        max_agreement_drop=args.gate_max_agreement_drop,
    )
    round_config = SelfLearningRoundConfig(
        num_candidate_positions=args.num_candidate_positions,
        selection_policy=args.selection_policy,
        engine_depth=args.engine_depth,
        stockfish_path=str(args.stockfish_path),
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        fine_tune_batch_size=args.fine_tune_batch_size,
        min_verified_examples=args.min_verified_examples,
        gate=gate,
        verifier=VerifierConfig(
            allowed_split="train",
            min_score_margin_cp=args.min_score_margin,
        ),
        feature_group=args.feature_group,
    )

    all_runs = []
    for index, seed in enumerate(seeds):
        baseline = args.baseline_checkpoint[0 if len(args.baseline_checkpoint) == 1 else index]
        run_config = RecursiveSelfLearningConfig(
            iterations=args.iterations,
            round_config=dataclasses.replace(round_config, seed=seed),
        )
        run_dir = args.output_dir / f"seed_{seed}"
        result = run_recursive_self_learning(
            baseline,
            train_records,
            dev_records,
            run_dir,
            run_config,
        )
        models = [
            {
                "name": "Original baseline",
                "role": "baseline",
                "checkpoint_path": str(baseline),
            },
            {
                "name": "Final accepted teacher",
                "role": "final_accepted",
                "checkpoint_path": result.final_checkpoint_path,
            },
        ]
        dev_summary = evaluate_model_suite(
            models, dev_records, manifest, "dev", warmup_runs=2, timed_runs=5
        )
        test_summary = evaluate_model_suite(
            models, valid_test_records, manifest, "test", warmup_runs=2, timed_runs=5
        )
        baseline_model, _ = load_checkpoint(baseline)
        final_model, _ = load_checkpoint(result.final_checkpoint_path)
        dev_examples = [_record_to_example(record) for record in dev_records]
        test_examples = [_record_to_example(record) for record in valid_test_records]
        dev_summary["regret_metrics"] = {
            "baseline": evaluate_regret_metrics(baseline_model, dev_examples),
            "final_accepted": evaluate_regret_metrics(final_model, dev_examples),
        }
        test_summary["regret_metrics"] = {
            "baseline": evaluate_regret_metrics(baseline_model, test_examples),
            "final_accepted": evaluate_regret_metrics(final_model, test_examples),
        }
        (run_dir / "final_dev_results.json").write_text(
            json.dumps(dev_summary, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "final_test_results.json").write_text(
            json.dumps(test_summary, indent=2) + "\n", encoding="utf-8"
        )
        all_runs.append(
            {
                "seed": seed,
                "baseline_checkpoint": str(baseline),
                "final_checkpoint": result.final_checkpoint_path,
                "recursive_summary": result.summary,
                "dev_results": dev_summary,
                "test_results": test_summary,
                "invalid_test_position_ids": invalid_test_ids,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recursive_experiment_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "train_records": len(train_records),
                "dev_records": len(dev_records),
                "test_records": len(valid_test_records),
                "invalid_test_records_excluded": len(invalid_test_ids),
                "runs": all_runs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
