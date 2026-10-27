"""Training pipeline for real Stockfish-labeled PositionRecord datasets.

This module trains an MLPRanker on labeled PositionRecords loaded from JSONL
files (produced by the labeling pipeline) rather than on synthetically
generated positions.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch

from .data import PositionRecord
from .distillation import _record_to_example
from .model import MLPRanker, ModelConfig
from .train import (
    TrainConfig,
    evaluate,
    make_targets,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


@dataclass
class RealTrainConfig:
    """Training configuration for real-data baseline (no synthetic-data fields)."""

    seed: int = 0
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 4
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0
    feature_group: str = "D"
    loss_mode: str = "classification"
    ranking_margin_scale: float = 100.0
    classification_weight: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    def to_train_config(self) -> TrainConfig:
        """Create a TrainConfig with dummy synthetic-data fields for checkpoint compatibility."""
        return TrainConfig(
            seed=self.seed,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout,
            # Dummy values for fields unused in real-data training.
            engine_depth=0,
            num_games=0,
            max_plies=0,
            stockfish_path="",
            feature_group=self.feature_group,
        )


def run_training_on_records(
    train_records: List[PositionRecord],
    dev_records: List[PositionRecord],
    config: RealTrainConfig,
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> dict:
    """Train an MLPRanker on real labeled PositionRecords.

    Parameters
    ----------
    train_records:
        Records with ``split == "train"`` and Stockfish scores populated.
    dev_records:
        Records with ``split == "dev"`` used for evaluation (not training).
    config:
        Training hyper-parameters.
    output_dir:
        Directory where checkpoints and logs are saved.

    Returns
    -------
    Dict with keys ``history``, ``split_info``, and ``output_dir``.

    Raises
    ------
    ValueError
        If any train_records have ``split != "train"``, or any dev_records
        have ``split != "dev"``, or dev/test records are present in
        train_records.
    """
    # --- Validation -----------------------------------------------------------
    bad_train = [r for r in train_records if r.split != "train"]
    if bad_train:
        raise ValueError(
            f"All train_records must have split='train'; "
            f"found {len(bad_train)} with other splits: "
            f"{[r.position_id for r in bad_train[:3]]}"
        )

    bad_dev = [r for r in dev_records if r.split != "dev"]
    if bad_dev:
        raise ValueError(
            f"All dev_records must have split='dev'; "
            f"found {len(bad_dev)} with other splits: "
            f"{[r.position_id for r in bad_dev[:3]]}"
        )

    # Extra safety: no dev or test records leaked into training set.
    non_train = [r for r in train_records if r.split != "train"]
    if non_train:
        raise ValueError(
            "dev/test records found in train_records: "
            + str([r.position_id for r in non_train[:3]])
        )

    # --- Setup ----------------------------------------------------------------
    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert records to examples ------------------------------------------
    train_examples = []
    train_skip_count = 0
    for rec in train_records:
        try:
            ex = _record_to_example(rec, config.feature_group)
            ex.target_index = make_targets(ex)
            train_examples.append(ex)
        except (ValueError, Exception):
            train_skip_count += 1

    dev_examples = []
    dev_skip_count = 0
    for rec in dev_records:
        try:
            ex = _record_to_example(rec, config.feature_group)
            ex.target_index = make_targets(ex)
            dev_examples.append(ex)
        except (ValueError, Exception):
            dev_skip_count += 1

    split_info = {
        "num_train_records": len(train_records),
        "num_dev_records": len(dev_records),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(dev_examples),
        "train_skipped": train_skip_count,
        "dev_skipped": dev_skip_count,
        "seed": config.seed,
    }

    # --- Model ----------------------------------------------------------------
    model_config = ModelConfig.for_feature_group(
        config.feature_group,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
    )
    model = MLPRanker(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_config_compat = config.to_train_config()

    # --- Training loop --------------------------------------------------------
    train_rng = random.Random(config.seed)
    history = []
    best_dev_regret = float("inf")
    best_ckpt = output_dir / "best.pt"
    started_at = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_examples,
            optimizer,
            config.batch_size,
            rng=train_rng,
            loss_mode=config.loss_mode,
            ranking_margin_scale=config.ranking_margin_scale,
            classification_weight=config.classification_weight,
        )
        if dev_examples:
            dev_metrics = evaluate(model, dev_examples)
            regrets = []
            non_mate_regrets = []
            for ex in dev_examples:
                with torch.no_grad():
                    pred = int(
                        torch.argmax(
                            model(
                                torch.from_numpy(ex.position_features).float(),
                                torch.from_numpy(ex.move_features).float(),
                            )
                        ).item()
                    )
                regret = ex.candidate_scores[ex.target_index] - ex.candidate_scores[pred]
                regrets.append(regret)
                if not any(abs(score) >= 90000 for score in ex.candidate_scores):
                    non_mate_regrets.append(regret)
            dev_metrics.update(
                {
                    "median_regret_cp": float(statistics.median(regrets)) if regrets else 0.0,
                    "non_mate_regret_cp": (
                        float(sum(non_mate_regrets) / len(non_mate_regrets))
                        if non_mate_regrets
                        else 0.0
                    ),
                    "non_mate_num_examples": len(non_mate_regrets),
                }
            )
        else:
            dev_metrics = {
                "loss": float("inf"),
                "top1_agreement": 0.0,
                "mean_regret_cp": 0.0,
                "median_regret_cp": 0.0,
                "non_mate_regret_cp": 0.0,
                "non_mate_num_examples": 0,
                "num_examples": 0,
            }

        epoch_seconds = time.perf_counter() - epoch_started_at
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_seconds": epoch_seconds,
                **dev_metrics,
            }
        )

        if dev_metrics["non_mate_regret_cp"] < best_dev_regret:
            best_dev_regret = dev_metrics["non_mate_regret_cp"]
            save_checkpoint(
                best_ckpt,
                model,
                train_config_compat,
                split_info,
                {"epoch": epoch, **dev_metrics},
            )

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_top1={dev_metrics['top1_agreement']:.3f} "
            f"dev_regret={dev_metrics['mean_regret_cp']:.1f} "
            f"dev_non_mate_regret={dev_metrics['non_mate_regret_cp']:.1f} "
            f"dev_median_regret={dev_metrics['median_regret_cp']:.1f} "
            f"epoch_seconds={epoch_seconds:.1f}"
        )

    # Save final checkpoint.
    final_ckpt = output_dir / "final.pt"
    save_checkpoint(final_ckpt, model, train_config_compat, split_info, {"history": history})

    # Persist logs.
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    config_data = {**config.to_dict(), **(metadata or {})}
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    best_metrics = (
        min(history, key=lambda item: item["non_mate_regret_cp"]) if history else {}
    )
    best_size = best_ckpt.stat().st_size if best_ckpt.exists() else 0
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_metrics.get("epoch"),
                "dev": {
                    key: best_metrics.get(key)
                    for key in (
                        "top1_agreement",
                        "mean_regret_cp",
                        "median_regret_cp",
                        "non_mate_regret_cp",
                        "loss",
                        "num_examples",
                        "non_mate_num_examples",
                    )
                },
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "best_checkpoint_size_bytes": best_size,
                "final_checkpoint_size_bytes": final_ckpt.stat().st_size
                if final_ckpt.exists()
                else 0,
                "feature_group": config.feature_group,
                "training_time_seconds": time.perf_counter() - started_at,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "history": history,
        "split_info": split_info,
        "output_dir": str(output_dir),
        "metrics": best_metrics,
    }


def load_records_from_jsonl(path: Path) -> List[PositionRecord]:
    """Load PositionRecords from a JSONL file. Returns empty list for empty files."""
    records: List[PositionRecord] = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(PositionRecord(**json.loads(line)))
    return records
