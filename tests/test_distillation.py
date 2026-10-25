from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chessevolve.distillation import (
    DistillationConfig,
    distillation_loss,
    probabilities,
    train_student,
    validate_records_against_manifest,
)
from chessevolve.data import PositionRecord
from chessevolve.dataset import TrainingExample
from chessevolve.encoding import MOVE_DIM, POSITION_DIM
from chessevolve.model import MLPRanker, ModelConfig


def _example() -> TrainingExample:
    return TrainingExample(
        position_features=np.zeros(POSITION_DIM, dtype=np.float32),
        move_features=np.zeros((4, MOVE_DIM), dtype=np.float32),
        candidate_scores=[1, 4, 2, 0],
        target_index=1,
        position_id="p",
        fen="startpos",
    )


def test_probabilities_normalized_and_temperature_positive():
    logits = torch.tensor([1.0, 2.0, -1.0])
    assert torch.allclose(probabilities(logits).sum(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="positive"):
        probabilities(logits, 0)
    for invalid in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            probabilities(logits, invalid)
    for invalid in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            DistillationConfig(temperature=invalid)


def test_distillation_loss_finite_and_gradients_flow():
    student = torch.randn(4, requires_grad=True)
    teacher = torch.randn(4)
    loss = distillation_loss(student, teacher, 2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    with pytest.raises(ValueError, match="positive"):
        distillation_loss(student, teacher, -1)


def test_training_outputs_normalized_teacher_student_and_student_is_smaller():
    teacher = MLPRanker()
    student = MLPRanker(ModelConfig(hidden_dim=8, num_hidden_layers=1))
    assert student.parameter_count() < teacher.parameter_count()
    example = _example()
    with torch.no_grad():
        teacher_probs = probabilities(teacher(
            torch.from_numpy(example.position_features),
            torch.from_numpy(example.move_features),
        ))
        student_probs = probabilities(student(
            torch.from_numpy(example.position_features),
            torch.from_numpy(example.move_features),
        ))
    assert torch.allclose(teacher_probs.sum(), torch.tensor(1.0))
    assert torch.allclose(student_probs.sum(), torch.tensor(1.0))
    history = train_student(
        student, teacher, [example],
        DistillationConfig(epochs=1, batch_size=1), "distilled"
    )
    assert len(history) == 1


def test_manifest_validation_preserves_train_and_dev_provenance(tmp_path: Path):
    manifest = {
        "source_game_ids": {"train": ["g_train"], "dev": ["g_dev"], "test": []}
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    train = PositionRecord(
        "train-1", "g_train", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ["e2e4"], "train", candidate_scores_cp=[10],
    )
    dev = PositionRecord(
        "dev-1", "g_dev", train.fen, ["e2e4"], "dev", candidate_scores_cp=[10],
    )
    assert validate_records_against_manifest([train], manifest_path, "train") == [train]
    assert validate_records_against_manifest([dev], manifest_path, "dev") == [dev]
    with pytest.raises(ValueError, match="expected 'train'"):
        validate_records_against_manifest([dev], manifest_path, "train")
