"""Baseline training pipeline for the Stockfish-supervised MLP ranker."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import PositionRecord
from .dataset import TrainingExample, build_examples, generate_random_positions, split_records
from .encoding import POSITION_DIM, MOVE_DIM
from .engine import EngineSettings, StockfishAdapter
from .model import MLPRanker, ModelConfig
from .splitting import SplitResult, write_split_manifest


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 4
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0
    engine_depth: int = 6
    num_games: int = 20
    max_plies: int = 16
    stockfish_path: str = "tools/stockfish/stockfish/stockfish-linux-x86-64-universal"

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_targets(example: TrainingExample) -> int:
    """Target = index of the Stockfish-best candidate (argmax of cp scores).

    Mate scores are already converted to +/- MATE_SCORE by the adapter, so
    argmax is consistent with Stockfish's perspective (side to move).
    """
    scores = example.candidate_scores
    return int(max(range(len(scores)), key=lambda i: scores[i]))  # type: ignore[arg-type]


def train_one_epoch(
    model: MLPRanker,
    examples: List[TrainingExample],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: Optional[random.Random] = None,
) -> float:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not examples:
        return 0.0

    model.train()
    total_loss = 0.0
    n = 0

    if rng is not None:
        examples = list(examples)
        rng.shuffle(examples)

    # Accumulate gradients over chunks of size up to batch_size.
    # The gradients for each update are normalized by the actual number of
    # examples in that update (including the final partial batch).
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        actual_batch_size = len(batch)
        optimizer.zero_grad()
        for ex in batch:
            pos = torch.from_numpy(ex.position_features).float()
            moves = torch.from_numpy(ex.move_features).float()
            target = torch.tensor(ex.target_index, dtype=torch.long)
            logits = model(pos, moves)  # (N,)
            loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            (loss / actual_batch_size).backward()
            total_loss += float(loss.item())
            n += 1
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: MLPRanker,
    examples: List[TrainingExample],
) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    top1 = 0
    regrets = []
    for ex in examples:
        pos = torch.from_numpy(ex.position_features).float()
        moves = torch.from_numpy(ex.move_features).float()
        target = torch.tensor(ex.target_index, dtype=torch.long)
        logits = model(pos, moves)
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        total_loss += float(loss.item())
        n += 1
        pred = int(torch.argmax(logits).item())
        if pred == ex.target_index:
            top1 += 1
        # regret in centipawns: best score - selected score
        best_score = ex.candidate_scores[ex.target_index]
        sel_score = ex.candidate_scores[pred]
        regrets.append(best_score - sel_score)  # type: ignore[operator]
    return {
        "loss": total_loss / max(n, 1),
        "top1_agreement": top1 / max(n, 1),
        "mean_regret_cp": float(np.mean(regrets)) if regrets else 0.0,
        "num_examples": n,
    }


def save_checkpoint(
    path: Path,
    model: MLPRanker,
    config: TrainConfig,
    split_info: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": config.to_dict(),
            "split_info": split_info,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path) -> Tuple[MLPRanker, dict]:
    ckpt = torch.load(path, map_location="cpu")
    mc = ModelConfig(**ckpt["model_config"])
    model = MLPRanker(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run_training(config: TrainConfig, output_dir: Path) -> dict:
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data.
    records = generate_random_positions(config.num_games, config.max_plies, config.seed)
    split_result = split_records(records, seed=config.seed)
    train_records = [r for r in split_result["train"]]
    dev_records = [r for r in split_result["dev"]]
    test_records = [r for r in split_result["test"]]

    settings = EngineSettings(depth=config.engine_depth)
    with StockfishAdapter(Path(config.stockfish_path)) as train_adapter:
        train_adapter.reset()
        train_examples, train_skipped = build_examples(train_records, train_adapter, settings)
    with StockfishAdapter(Path(config.stockfish_path)) as dev_adapter:
        dev_adapter.reset()
        dev_examples, dev_skipped = build_examples(dev_records, dev_adapter, settings)

    # Fix target_index alignment (already set in build_examples; recompute to be safe).
    for ex in train_examples + dev_examples:
        ex.target_index = make_targets(ex)

    model_config = ModelConfig(
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        dropout=config.dropout,
    )
    model = MLPRanker(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    split_info = {
        "num_train_records": len(train_records),
        "num_dev_records": len(dev_records),
        "num_test_records": len(test_records),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(dev_examples),
        "train_skipped": train_skipped,
        "dev_skipped": dev_skipped,
        "seed": config.seed,
    }

    train_rng = random.Random(config.seed)
    history = []
    best_dev_loss = float("inf")
    best_ckpt = output_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_examples,
            optimizer,
            config.batch_size,
            rng=train_rng,
        )
        dev_metrics = evaluate(model, dev_examples)
        history.append({"epoch": epoch, "train_loss": train_loss, **dev_metrics})
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            save_checkpoint(best_ckpt, model, config, split_info, {"epoch": epoch, **dev_metrics})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_metrics['loss']:.4f} dev_top1={dev_metrics['top1_agreement']:.3f} dev_regret={dev_metrics['mean_regret_cp']:.1f}")

    # Save final checkpoint.
    final_ckpt = output_dir / "final.pt"
    save_checkpoint(final_ckpt, model, config, split_info, {"history": history})
    # Save history + config.
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (output_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    # Persist evaluated datasets and manifest
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    evaluated_train_records = [ex.record for ex in train_examples if ex.record is not None]
    evaluated_dev_records = [ex.record for ex in dev_examples if ex.record is not None]

    import dataclasses
    def _save_records_jsonl(path: Path, recs: List[PositionRecord]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for r in recs:
                handle.write(json.dumps(dataclasses.asdict(r)) + "\n")

    _save_records_jsonl(dataset_dir / "train_records.jsonl", evaluated_train_records)
    _save_records_jsonl(dataset_dir / "dev_records.jsonl", evaluated_dev_records)

    write_split_manifest(
        dataset_dir / "manifest.json",
        {"train": evaluated_train_records, "dev": evaluated_dev_records, "test": test_records},
        seed=config.seed,
        proportions={"train": 0.7, "dev": 0.15, "test": 0.15},
    )

    return {"history": history, "split_info": split_info, "output_dir": str(output_dir)}
