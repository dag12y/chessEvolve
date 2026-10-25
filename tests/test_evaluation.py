"""Tests for Milestone 5 evaluation module, metrics, and isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chessevolve.data import PositionRecord
from chessevolve.dataset import TrainingExample
from chessevolve.encoding import MOVE_DIM, POSITION_DIM
from chessevolve.evaluation import (
    compute_file_sha256,
    evaluate_checkpoint,
    evaluate_model_performance,
    get_model_metadata,
    measure_inference_latency,
    records_to_evaluation_examples,
    save_evaluation_artifacts,
)
from chessevolve.model import MLPRanker, ModelConfig
from chessevolve.train import evaluate as baseline_evaluate


def _create_mock_example(
    scores: list[int],
    position_id: str = "mock-pos-1",
    target_idx: int | None = None,
) -> TrainingExample:
    """Create a mock training example with defined candidate scores."""
    k = len(scores)
    target = target_idx if target_idx is not None else int(np.argmax(scores))
    return TrainingExample(
        position_features=np.zeros(POSITION_DIM, dtype=np.float32),
        move_features=np.zeros((k, MOVE_DIM), dtype=np.float32),
        candidate_scores=scores,
        target_index=target,
        position_id=position_id,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )


def test_consistent_metric_computation():
    """Verify top-1 agreement and mean regret cp match exact arithmetic and M1/M2 API."""
    model = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))
    model.eval()

    ex1 = _create_mock_example([100, 50, 0], position_id="ex1", target_idx=0)
    ex2 = _create_mock_example([20, 80, 10], position_id="ex2", target_idx=1)
    examples = [ex1, ex2]

    metrics = evaluate_model_performance(model, examples)
    baseline_metrics = baseline_evaluate(model, examples)

    assert metrics["num_examples"] == 2
    assert metrics["top1_agreement"] == baseline_metrics["top1_agreement"]
    assert metrics["mean_regret_cp"] == baseline_metrics["mean_regret_cp"]
    assert np.isclose(metrics["loss"], baseline_metrics["loss"])
    assert metrics["mean_regret_cp"] >= 0.0


def test_empty_evaluation_data_safely_handled():
    """Verify empty dataset returns safe zero-valued metrics without zero-division error."""
    model = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))
    metrics = evaluate_model_performance(model, [])
    assert metrics["num_examples"] == 0
    assert metrics["top1_agreement"] == 0.0
    assert metrics["mean_regret_cp"] == 0.0
    assert metrics["loss"] == 0.0

    latency = measure_inference_latency(model, [])
    assert latency["latency_ms_per_pos_median"] == 0.0


def test_invalid_and_non_finite_outputs_safely_handled():
    """Verify non-finite model weights/logits raise FloatingPointError."""
    model = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))
    # Inject NaN into model parameter
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
            break

    ex = _create_mock_example([10, 20])
    with pytest.raises(FloatingPointError, match="Non-finite logits"):
        evaluate_model_performance(model, [ex])


def test_model_output_and_candidate_count_consistency():
    """Verify ranking works across positions with varying candidate move counts."""
    model = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))
    examples = [
        _create_mock_example([10, 20], position_id="2-moves"),
        _create_mock_example([5, 15, 25, 35, 45], position_id="5-moves"),
        _create_mock_example(list(range(28)), position_id="28-moves"),
    ]
    metrics = evaluate_model_performance(model, examples)
    assert metrics["num_examples"] == 3
    assert 0.0 <= metrics["top1_agreement"] <= 1.0


def test_checkpoint_metadata_and_comparison(tmp_path: Path):
    """Verify get_model_metadata extracts parameter counts, size, and sha256."""
    teacher = MLPRanker(ModelConfig(hidden_dim=64, num_hidden_layers=2))
    student = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))

    t_path = tmp_path / "teacher.pt"
    s_path = tmp_path / "student.pt"

    torch.save(
        {"model_state_dict": teacher.state_dict(), "model_config": teacher.config.to_dict()},
        t_path,
    )
    torch.save(
        {"model_state_dict": student.state_dict(), "model_config": student.config.to_dict()},
        s_path,
    )

    t_meta = get_model_metadata(t_path)
    s_meta = get_model_metadata(s_path)

    assert t_meta["trainable_parameters"] > s_meta["trainable_parameters"]
    assert t_meta["checkpoint_size_bytes"] > 0
    assert len(t_meta["checkpoint_sha256"]) == 64
    assert compute_file_sha256(t_path) == t_meta["checkpoint_sha256"]


def test_latency_measurement_benchmarking():
    """Verify inference latency benchmarking returns positive metrics."""
    model = MLPRanker(ModelConfig(hidden_dim=16, num_hidden_layers=1))
    examples = [_create_mock_example([10, 20, 30], position_id=f"pos-{i}") for i in range(3)]

    lat = measure_inference_latency(model, examples, warmup_runs=2, timed_runs=5)
    assert lat["latency_ms_per_pos_median"] > 0.0
    assert lat["latency_ms_per_pos_mean"] > 0.0
    assert lat["warmup_runs"] == 2
    assert lat["timed_runs"] == 5


def test_final_test_isolation_enforced(tmp_path: Path):
    """Verify split isolation strictly prevents test leakage into dev evaluation."""
    rec_train = PositionRecord(
        position_id="train-1",
        source_game_id="game-train",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        candidate_moves=["e2e4", "d2d4"],
        split="train",
        candidate_scores_cp=[10, 20],
    )
    rec_dev = PositionRecord(
        position_id="dev-1",
        source_game_id="game-dev",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        candidate_moves=["e2e4", "d2d4"],
        split="dev",
        candidate_scores_cp=[10, 20],
    )
    rec_test = PositionRecord(
        position_id="test-1",
        source_game_id="game-test",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        candidate_moves=["e2e4", "d2d4"],
        split="test",
        candidate_scores_cp=[10, 20],
    )

    # records_to_evaluation_examples must reject test records when split is dev
    with pytest.raises(ValueError, match="does not match expected split 'dev'"):
        records_to_evaluation_examples([rec_test], expected_split="dev")

    # records_to_evaluation_examples must reject dev records when split is test
    with pytest.raises(ValueError, match="does not match expected split 'test'"):
        records_to_evaluation_examples([rec_dev], expected_split="test")

    # Valid conversion for matching split
    dev_exs = records_to_evaluation_examples([rec_dev], expected_split="dev")
    assert len(dev_exs) == 1
    assert dev_exs[0].position_id == "dev-1"


def test_save_evaluation_artifacts_creates_files(tmp_path: Path):
    """Verify JSON, CSV, and PNG charts are properly generated and saved."""
    summary = {
        "timestamp_utc": "2026-10-25T23:00:00Z",
        "split_name": "dev",
        "is_final_test": False,
        "num_positions": 2,
        "models": [
            {
                "model_name": "Baseline Teacher",
                "role": "baseline_teacher",
                "split": "dev",
                "checkpoint_path": str(tmp_path / "dummy.pt"),
                "checkpoint_sha256": "abc123",
                "checkpoint_size_bytes": 1024,
                "checkpoint_size_kib": 1.0,
                "trainable_parameters": 1000,
                "total_parameters": 1000,
                "num_evaluated_positions": 2,
                "top1_agreement": 0.5,
                "mean_regret_cp": 15.0,
                "loss": 2.5,
                "latency_ms_per_pos_median": 0.05,
                "latency_ms_per_pos_mean": 0.05,
                "latency_ms_per_pos_std": 0.001,
                "notes": "",
            }
        ],
    }

    out_dir = tmp_path / "results"
    json_path, csv_path, charts = save_evaluation_artifacts(summary, out_dir)

    assert json_path.is_file()
    assert csv_path.is_file()
    assert len(charts) == 2
    for c in charts:
        assert c.is_file()
        assert c.stat().st_size > 0
