"""CPU-friendly hard-label and teacher-distillation training for M4."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import PositionRecord
from .dataset import TrainingExample, build_examples
from .encoding import encode_move, encode_position
from .engine import EngineSettings, StockfishAdapter
from .model import MLPRanker, ModelConfig
from .train import evaluate, load_checkpoint, make_targets


@dataclass
class DistillationConfig:
    seed: int = 0
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 4
    temperature: float = 2.0
    hard_loss_weight: float = 0.0
    hidden_dim: int = 32
    num_hidden_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if not 0 <= self.hard_loss_weight <= 1:
            raise ValueError("hard_loss_weight must be between 0 and 1")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return a normalized, finite probability vector."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be a finite positive number")
    return torch.softmax(logits / temperature, dim=-1)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    target_index: Optional[torch.Tensor] = None,
    hard_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Stable temperature-scaled KL loss, optionally mixed with CE."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be a finite positive number")
    if not 0 <= hard_loss_weight <= 1:
        raise ValueError("hard_loss_weight must be between 0 and 1")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        teacher_probs = probabilities(teacher_logits, temperature)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    loss = loss * (temperature * temperature)
    if target_index is not None and hard_loss_weight:
        hard = F.cross_entropy(student_logits.unsqueeze(0), target_index.unsqueeze(0))
        loss = (1 - hard_loss_weight) * loss + hard_loss_weight * hard
    return loss


def _record_to_example(record: PositionRecord) -> TrainingExample:
    import chess

    board = chess.Board(record.fen)
    moves = [chess.Move.from_uci(move) for move in record.candidate_moves]
    if any(move not in board.legal_moves for move in moves):
        raise ValueError(f"illegal candidate in record {record.position_id}")
    if record.candidate_scores_cp is None or any(
        score is None for score in record.candidate_scores_cp
    ):
        raise ValueError(f"record {record.position_id} has no reference scores")
    scores = [int(score) for score in record.candidate_scores_cp]  # type: ignore[arg-type]
    return TrainingExample(
        position_features=encode_position(board),
        move_features=np.stack([encode_move(board, move) for move in moves]),
        candidate_scores=scores,
        target_index=int(np.argmax(scores)),
        position_id=record.position_id,
        fen=record.fen,
        record=record,
    )


def records_to_examples(records: Iterable[PositionRecord]) -> List[TrainingExample]:
    """Convert only eligible records; non-training records are rejected."""
    examples = []
    for record in records:
        if record.split != "train":
            raise ValueError(
                f"student training requires train records, got {record.split!r}"
            )
        examples.append(_record_to_example(record))
    return examples


def load_records(path: Path) -> List[PositionRecord]:
    """Load JSONL records from a file or a baseline experiment directory."""
    if path.is_dir():
        path = path / "train_records.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(PositionRecord(**json.loads(line)))
    return records


def validate_records_against_manifest(
    records: Iterable[PositionRecord], manifest_path: Path, expected_split: str
) -> List[PositionRecord]:
    """Validate record split and source-game membership against a frozen manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "source_game_ids" not in manifest:
        raise ValueError(f"Invalid split manifest: {manifest_path}")
    declared_games = set(manifest["source_game_ids"].get(expected_split, []))
    validated = list(records)
    if not validated:
        raise ValueError(f"No {expected_split} records were loaded")
    for record in validated:
        if record.split != expected_split:
            raise ValueError(
                f"Record {record.position_id} has split {record.split!r}, "
                f"expected {expected_split!r}"
            )
        if record.source_game_id not in declared_games:
            raise ValueError(
                f"Record {record.position_id} source game {record.source_game_id!r} "
                f"is not declared in the {expected_split!r} manifest split"
            )
    return validated


def train_student(
    student: MLPRanker,
    teacher: MLPRanker,
    examples: List[TrainingExample],
    config: DistillationConfig,
    method: str,
) -> List[dict]:
    if method not in {"hard", "distilled"}:
        raise ValueError(f"unknown student method: {method}")
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
    rng = random.Random(config.seed)
    history = []
    teacher.eval()
    for epoch in range(1, config.epochs + 1):
        shuffled = list(examples)
        rng.shuffle(shuffled)
        student.train()
        total = 0.0
        for start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[start : start + config.batch_size]
            optimizer.zero_grad()
            losses = []
            for example in batch:
                pos = torch.from_numpy(example.position_features).float()
                moves = torch.from_numpy(example.move_features).float()
                target = torch.tensor(example.target_index, dtype=torch.long)
                student_logits = student(pos, moves)
                if method == "hard":
                    loss = F.cross_entropy(
                        student_logits.unsqueeze(0), target.unsqueeze(0)
                    )
                else:
                    with torch.no_grad():
                        teacher_logits = teacher(pos, moves)
                    loss = distillation_loss(
                        student_logits,
                        teacher_logits,
                        config.temperature,
                        target,
                        config.hard_loss_weight,
                    )
                losses.append(loss)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {method} loss")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        history.append({"epoch": epoch, "train_loss": total / max(len(examples), 1)})
    return history


def save_student_checkpoint(
    path: Path,
    student: MLPRanker,
    teacher_checkpoint: Path,
    config: DistillationConfig,
    method: str,
    history: List[dict],
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(
            {
                "model_state_dict": student.state_dict(),
                "model_config": student.config.to_dict(),
                "distillation_config": config.to_dict(),
                "method": method,
                "teacher_checkpoint": str(teacher_checkpoint),
                "history": history,
                "metrics": metrics,
                "parameter_count": student.parameter_count(),
            },
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_student(
    teacher_checkpoint: Path,
    records: List[PositionRecord],
    output_dir: Path,
    config: DistillationConfig,
    method: str,
    dev_records: Optional[List[PositionRecord]] = None,
) -> dict:
    set_seed(config.seed)
    teacher, teacher_metadata = load_checkpoint(teacher_checkpoint)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    examples = records_to_examples(records)
    student = MLPRanker(
        ModelConfig(
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )
    )
    if student.parameter_count() >= teacher.parameter_count():
        raise ValueError("student must have fewer parameters than teacher")
    history = train_student(student, teacher, examples, config, method)
    train_metrics = evaluate(student, examples)
    dev_metrics = None
    if dev_records is not None:
        if any(record.split != "dev" for record in dev_records):
            raise ValueError("development evaluation requires dev records")
        dev_examples = [_record_to_example(record) for record in dev_records]
        dev_metrics = evaluate(student, dev_examples)
    metrics = dict(dev_metrics or train_metrics)
    metrics.update(
        {
            "method": method,
            "seed": config.seed,
            "num_train_examples": len(examples),
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "teacher_parameter_count": teacher.parameter_count(),
            "student_parameter_count": student.parameter_count(),
            "teacher_checkpoint_metrics": teacher_metadata.get("metrics", {}),
        }
    )
    checkpoint = output_dir / f"{method}_seed{config.seed}.pt"
    save_student_checkpoint(
        checkpoint, student, teacher_checkpoint, config, method, history, metrics
    )
    (output_dir / f"{method}_seed{config.seed}.json").write_text(
        json.dumps({"config": config.to_dict(), "metrics": metrics, "history": history}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"checkpoint": str(checkpoint), "metrics": metrics, "history": history}
