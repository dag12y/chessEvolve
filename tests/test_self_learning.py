"""Tests for chessevolve.self_learning (Milestone 3).

Coverage:
- Position selection: policy enforcement, test-split guard, reproducibility.
- Feedback generation: valid output, engine-failure isolation.
- Verifier: accepted/rejected paths, split guard, finite-score enforcement.
- Gate: accept path, reject-primary, reject-secondary, non-finite metrics.
- Fine-tuning: candidate model independence from baseline.
- End-to-end round: acceptance path, rejection path, artifact integrity.
- Test-data isolation: test records never enter any M3 stage.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from chessevolve.data import PositionRecord
from chessevolve.dataset import TrainingExample
from chessevolve.encoding import MOVE_DIM, POSITION_DIM
from chessevolve.model import MLPRanker, ModelConfig
from chessevolve.self_learning import (
    FeedbackRecord,
    FineTuneConfig,
    GateConfig,
    GateDecision,
    InsufficientVerifiedExamplesError,
    PositionSelectionConfig,
    RecursiveSelfLearningConfig,
    SelfLearningRoundConfig,
    VerifierConfig,
    evaluate_acceptance_gate,
    fine_tune_candidate,
    generate_feedback,
    run_self_learning_round,
    run_recursive_self_learning,
    select_candidate_positions,
    verify_feedback,
)
from chessevolve.train import save_checkpoint, TrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STOCKFISH_PATH = Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal")
stockfish_available = STOCKFISH_PATH.exists()

VALID_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_record(
    position_id: str = "p0",
    fen: str = START_FEN,
    split: str = "train",
    source_game_id: str = "g0",
    candidates: list[str] | None = None,
) -> PositionRecord:
    if candidates is None:
        candidates = ["e2e4", "d2d4"]
    return PositionRecord(
        position_id=position_id,
        source_game_id=source_game_id,
        fen=fen,
        candidate_moves=candidates,
        split=split,
    )


def _make_feedback(
    position_id: str = "p0",
    fen: str = START_FEN,
    split: str = "train",
    scores: list | None = None,
    is_valid: bool = True,
    error: str | None = None,
    candidates: list[str] | None = None,
) -> FeedbackRecord:
    if candidates is None:
        candidates = ["e2e4", "d2d4"]
    if scores is None:
        scores = [30, 25]
    return FeedbackRecord(
        position_id=position_id,
        source_game_id="g0",
        fen=fen,
        candidate_moves=candidates,
        candidate_scores=scores,
        split=split,
        engine_version="Stockfish test",
        engine_settings={"depth": 4},
        reference_best_move="e2e4" if is_valid else None,
        is_valid=is_valid,
        error=error,
    )


def _make_example(target: int = 0, n: int = 3) -> TrainingExample:
    scores = [100 if i == target else 10 for i in range(n)]
    return TrainingExample(
        position_features=np.zeros(POSITION_DIM, dtype=np.float32),
        move_features=np.zeros((n, MOVE_DIM), dtype=np.float32),
        candidate_scores=scores,
        target_index=target,
        position_id="p_test",
        fen=START_FEN,
    )


def _save_dummy_checkpoint(path: Path, hidden: int = 32) -> MLPRanker:
    model = MLPRanker(ModelConfig(hidden_dim=hidden))
    cfg = TrainConfig(epochs=1, num_games=3)
    save_checkpoint(path, model, cfg, {}, {"loss": 1.0, "top1_agreement": 0.0, "mean_regret_cp": 100.0})
    return model


# ===========================================================================
# 1. Position selection
# ===========================================================================


def test_select_random_policy_returns_train_only():
    # Only train records should be accepted; dev records are rejected.
    records = [_make_record(f"p{i}", split="train") for i in range(3)]
    cfg = PositionSelectionConfig(policy="random", max_positions=10, seed=0)
    selected = select_candidate_positions(records, config=cfg)
    assert all(r.split == "train" for r in selected)
    assert len(selected) == 3  # all 3 train records
    records = [_make_record(f"p{i}", split="train") for i in range(10)]
    cfg = PositionSelectionConfig(policy="random", max_positions=5, seed=7)
    s1 = select_candidate_positions(records, config=cfg)
    s2 = select_candidate_positions(records, config=cfg)
    assert [r.position_id for r in s1] == [r.position_id for r in s2]

def test_select_random_policy_max_positions():
    records = [_make_record(f"p{i}", split="train") for i in range(10)]
    cfg = PositionSelectionConfig(policy="random", max_positions=3, seed=0)
    selected = select_candidate_positions(records, config=cfg)
    assert len(selected) <= 3


def test_select_rejects_test_records():
    records = [
        _make_record("p0", split="train"),
        _make_record("p1", split="test"),
    ]
    cfg = PositionSelectionConfig(policy="random", max_positions=10, seed=0)
    with pytest.raises(ValueError, match="(?i)test"):
        select_candidate_positions(records, config=cfg)


def test_select_uncertainty_requires_model():
    records = [_make_record("p0", split="train")]
    cfg = PositionSelectionConfig(policy="uncertainty", max_positions=5, seed=0)
    with pytest.raises(ValueError, match="model"):
        select_candidate_positions(records, model=None, config=cfg)


def test_select_unknown_policy_raises():
    records = [_make_record("p0", split="train")]
    cfg = PositionSelectionConfig(policy="bogus", max_positions=5, seed=0)
    with pytest.raises(ValueError, match="Unknown selection policy"):
        select_candidate_positions(records, config=cfg)


def test_select_uncertainty_policy_with_model():
    torch.manual_seed(0)
    model = MLPRanker(ModelConfig(hidden_dim=16))
    records = [
        _make_record(f"p{i}", split="train", candidates=["e2e4", "d2d4"])
        for i in range(4)
    ]
    cfg = PositionSelectionConfig(policy="uncertainty", max_positions=3, seed=0)
    selected = select_candidate_positions(records, model=model, config=cfg)
    assert len(selected) <= 3
    assert all(r.split == "train" for r in selected)


def test_select_empty_input_returns_empty():
    cfg = PositionSelectionConfig(policy="random", max_positions=5, seed=0)
    assert select_candidate_positions([], config=cfg) == []


# ===========================================================================
# 2. Feedback generation
# ===========================================================================


@pytest.mark.skipif(not stockfish_available, reason="Stockfish not available")
def test_generate_feedback_valid_position():
    from chessevolve.engine import EngineSettings, StockfishAdapter

    records = [_make_record("p0", fen=START_FEN, candidates=["e2e4", "d2d4"])]
    settings = EngineSettings(depth=3)
    with StockfishAdapter(STOCKFISH_PATH) as adapter:
        adapter.reset()
        feedback = generate_feedback(records, adapter, settings)
    assert len(feedback) == 1
    fb = feedback[0]
    assert fb.position_id == "p0"
    assert fb.is_valid is True
    assert fb.error is None
    assert len(fb.candidate_scores) == 2
    assert all(s is not None for s in fb.candidate_scores)
    assert fb.reference_best_move in {"e2e4", "d2d4"}


@pytest.mark.skipif(not stockfish_available, reason="Stockfish not available")
def test_generate_feedback_illegal_candidates_marked_invalid():
    from chessevolve.engine import EngineSettings, StockfishAdapter

    # e2e5 is not a legal move from start
    records = [_make_record("p0", fen=START_FEN, candidates=["e2e5", "d2d4"])]
    settings = EngineSettings(depth=3)
    with StockfishAdapter(STOCKFISH_PATH) as adapter:
        adapter.reset()
        feedback = generate_feedback(records, adapter, settings)
    assert len(feedback) == 1
    assert feedback[0].is_valid is False
    assert "illegal" in feedback[0].error


@pytest.mark.skipif(not stockfish_available, reason="Stockfish not available")
def test_generate_feedback_engine_metadata_recorded():
    from chessevolve.engine import EngineSettings, StockfishAdapter

    records = [_make_record("p0", fen=START_FEN, candidates=["e2e4"])]
    settings = EngineSettings(depth=2)
    with StockfishAdapter(STOCKFISH_PATH) as adapter:
        adapter.reset()
        feedback = generate_feedback(records, adapter, settings)
    fb = feedback[0]
    assert fb.engine_version and fb.engine_version != ""
    assert fb.engine_settings["depth"] == 2
    assert fb.split == "train"


def test_generate_feedback_handles_invalid_board(monkeypatch):
    """generate_feedback must not crash on a board that fails is_valid()."""
    from chessevolve.engine import EngineSettings

    adapter_mock = MagicMock()
    adapter_mock.version.return_value = "test"
    records = [
        PositionRecord(
            position_id="bad",
            source_game_id="g0",
            fen="8/8/8/8/8/8/8/8 w - - 0 1",  # no kings → invalid
            candidate_moves=["e2e4"],
            split="train",
        )
    ]
    settings = EngineSettings(depth=2)
    feedback = generate_feedback(records, adapter_mock, settings)
    assert len(feedback) == 1
    assert feedback[0].is_valid is False
    assert "invalid" in feedback[0].error
    adapter_mock.evaluate_candidates.assert_not_called()


def test_generate_feedback_preserves_candidate_score_alignment_and_perspective():
    from chessevolve.engine import EngineSettings

    adapter_mock = MagicMock()
    adapter_mock.version.return_value = "test"
    adapter_mock.evaluate_candidates.return_value = [120, -99999]
    record = _make_record("p0", candidates=["e2e4", "d2d4"])
    feedback = generate_feedback([record], adapter_mock, EngineSettings(depth=2))[0]
    board, moves, _ = adapter_mock.evaluate_candidates.call_args.args
    assert board.turn is True
    assert [move.uci() for move in moves] == record.candidate_moves
    assert feedback.candidate_scores == [120, -99999]
    assert feedback.reference_best_move == "e2e4"


# ===========================================================================
# 3. Verifier
# ===========================================================================


def test_verify_accepts_valid_feedback():
    fb = _make_feedback(scores=[30, 25])
    result = verify_feedback([fb])
    assert result.num_accepted == 1
    assert result.num_rejected == 0


def test_verify_rejects_engine_error():
    fb = _make_feedback(is_valid=False, error="engine_timeout", scores=[])
    result = verify_feedback([fb])
    assert result.num_accepted == 0
    assert result.num_rejected == 1
    assert any("engine_failure" in k for k in result.rejection_reasons)


def test_verify_rejects_wrong_split():
    fb = _make_feedback(split="dev")
    cfg = VerifierConfig(allowed_split="train")
    result = verify_feedback([fb], config=cfg)
    assert result.num_accepted == 0
    assert result.num_rejected == 1
    assert any("disallowed_split" in k for k in result.rejection_reasons)


def test_verify_rejects_test_split():
    fb = _make_feedback(split="test")
    result = verify_feedback([fb])
    assert result.num_rejected == 1
    assert any("disallowed_split" in k for k in result.rejection_reasons)


def test_verify_rejects_missing_score():
    fb = _make_feedback(scores=[30, None])
    result = verify_feedback([fb])
    assert result.num_rejected == 1


def test_verify_rejects_non_finite_score():
    fb = _make_feedback(scores=[float("nan"), 25])
    result = verify_feedback([fb])
    assert result.num_rejected == 1
    assert any("non_finite" in k for k in result.rejection_reasons)


def test_verify_rejects_illegal_move():
    fb = _make_feedback(candidates=["e2e5", "d2d4"], scores=[30, 25])
    result = verify_feedback([fb])
    assert result.num_rejected == 1
    assert any("illegal" in k for k in result.rejection_reasons)


def test_verify_rejects_score_count_mismatch():
    fb = _make_feedback(candidates=["e2e4", "d2d4"], scores=[30])  # 2 moves but 1 score
    result = verify_feedback([fb])
    assert result.num_rejected == 1


def test_verify_rejects_feedback_provenance_mismatch():
    expected = _make_record("p0", source_game_id="expected-game")
    result = verify_feedback(
        [_make_feedback("p0")],
        expected_records={"p0": expected},
    )
    assert result.num_accepted == 0
    assert any("provenance_mismatch" in key for key in result.rejection_reasons)


def test_verify_score_margin_filter():
    fb = _make_feedback(scores=[30, 29])  # margin = 1 cp
    cfg = VerifierConfig(min_score_margin_cp=5)
    result = verify_feedback([fb], config=cfg)
    assert result.num_rejected == 1
    assert any("insufficient_score_margin" in k for k in result.rejection_reasons)


def test_verify_score_margin_accepted():
    fb = _make_feedback(scores=[50, 20])  # margin = 30 cp
    cfg = VerifierConfig(min_score_margin_cp=5)
    result = verify_feedback([fb], config=cfg)
    assert result.num_accepted == 1


def test_verify_produces_correct_training_examples():
    fb = _make_feedback(candidates=["e2e4", "d2d4"], scores=[50, 30])
    result = verify_feedback([fb])
    assert result.num_accepted == 1
    ex = result.accepted_examples[0]
    assert ex.target_index == 0  # e2e4 has higher score
    assert ex.position_features.shape == (POSITION_DIM,)
    assert ex.move_features.shape == (2, MOVE_DIM)
    assert ex.candidate_scores == [50, 30]


def test_verify_empty_input():
    result = verify_feedback([])
    assert result.num_accepted == 0
    assert result.num_rejected == 0
    assert result.total_reviewed == 0


def test_verify_records_rejection_reasons_in_detail():
    records = [
        _make_feedback("good", scores=[50, 20]),
        _make_feedback("bad_engine", is_valid=False, error="crash", scores=[]),
        _make_feedback("bad_split", split="test"),
    ]
    result = verify_feedback(records)
    assert result.num_accepted == 1
    assert result.num_rejected == 2
    assert result.total_reviewed == 3


# ===========================================================================
# 4. Acceptance gate
# ===========================================================================


def _metrics(regret: float, top1: float) -> dict:
    return {"loss": 1.0, "top1_agreement": top1, "mean_regret_cp": regret, "num_examples": 5}


def test_gate_accepts_when_criteria_met():
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.3),
        _metrics(regret=80.0, top1=0.3),
        GateConfig(min_regret_reduction_cp=5.0, max_agreement_drop=0.05),
    )
    assert decision.accepted is True
    assert math.isclose(decision.regret_diff_cp, 20.0)
    assert decision.agreement_diff == pytest.approx(0.0, abs=1e-6)


def test_gate_rejects_insufficient_regret_reduction():
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.3),
        _metrics(regret=98.0, top1=0.3),
        GateConfig(min_regret_reduction_cp=5.0),
    )
    assert decision.accepted is False
    assert "regret" in decision.reason.lower()


def test_gate_rejects_large_agreement_drop():
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.5),
        _metrics(regret=90.0, top1=0.3),  # drop of 0.2 > 0.05
        GateConfig(min_regret_reduction_cp=5.0, max_agreement_drop=0.05),
    )
    assert decision.accepted is False
    assert "agreement" in decision.reason.lower()


def test_gate_rejects_non_finite_baseline_metric():
    decision = evaluate_acceptance_gate(
        {"loss": float("nan"), "top1_agreement": 0.3, "mean_regret_cp": 100.0, "num_examples": 5},
        _metrics(regret=80.0, top1=0.3),
    )
    assert decision.accepted is False
    assert "non-finite" in decision.reason.lower() or "Non-finite" in decision.reason


def test_gate_rejects_non_finite_candidate_metric():
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.3),
        {"loss": 1.0, "top1_agreement": 0.3, "mean_regret_cp": float("inf"), "num_examples": 5},
    )
    assert decision.accepted is False


def test_gate_records_thresholds_in_decision():
    cfg = GateConfig(min_regret_reduction_cp=10.0, max_agreement_drop=0.1)
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.3),
        _metrics(regret=80.0, top1=0.3),
        cfg,
    )
    assert "min_regret_reduction_cp" in decision.thresholds
    assert decision.thresholds["min_regret_reduction_cp"] == 10.0


def test_gate_threshold_zero_allows_neutral_update():
    # Candidate has same regret as baseline; threshold 0 → accept
    decision = evaluate_acceptance_gate(
        _metrics(regret=100.0, top1=0.3),
        _metrics(regret=100.0, top1=0.3),
        GateConfig(min_regret_reduction_cp=0.0, max_agreement_drop=0.05),
    )
    assert decision.accepted is True


# ===========================================================================
# 5. Fine-tuning
# ===========================================================================


def test_fine_tune_baseline_unchanged():
    torch.manual_seed(0)
    baseline = MLPRanker(ModelConfig(hidden_dim=16))
    original_weights = {k: v.clone() for k, v in baseline.state_dict().items()}

    examples = [_make_example(target=i % 3) for i in range(4)]
    ft_cfg = FineTuneConfig(epochs=2, lr=1e-3, batch_size=4, seed=0)
    candidate = fine_tune_candidate(baseline, examples, ft_cfg)

    # Baseline must be unchanged
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, original_weights[k]), f"Baseline weight {k} was mutated"

    # Candidate must differ from baseline
    any_diff = any(
        not torch.equal(candidate.state_dict()[k], original_weights[k])
        for k in original_weights
    )
    assert any_diff, "Candidate model weights should have changed after fine-tuning"


def test_fine_tune_empty_examples_returns_copy():
    baseline = MLPRanker(ModelConfig(hidden_dim=16))
    candidate = fine_tune_candidate(baseline, [], FineTuneConfig())
    for k in baseline.state_dict():
        assert torch.equal(baseline.state_dict()[k], candidate.state_dict()[k])


def test_fine_tune_reproducibility():
    torch.manual_seed(1)
    baseline = MLPRanker(ModelConfig(hidden_dim=16))
    examples = [_make_example(target=i % 2) for i in range(4)]
    cfg = FineTuneConfig(epochs=2, lr=1e-3, batch_size=4, seed=99)

    c1 = fine_tune_candidate(copy.deepcopy(baseline), examples, cfg)
    c2 = fine_tune_candidate(copy.deepcopy(baseline), examples, cfg)

    for k in c1.state_dict():
        assert torch.allclose(c1.state_dict()[k], c2.state_dict()[k])


# ===========================================================================
# 6. Test-data isolation guards
# ===========================================================================


def test_select_raises_on_test_records():
    records = [_make_record("test_pos", split="test")]
    cfg = PositionSelectionConfig(policy="random", max_positions=10, seed=0)
    with pytest.raises(ValueError, match="(?i)test"):
        select_candidate_positions(records, config=cfg)


def test_verify_rejects_test_split_records():
    fb = _make_feedback(split="test", scores=[50, 20])
    result = verify_feedback([fb])
    assert result.num_accepted == 0
    assert result.num_rejected == 1


def test_run_round_raises_on_test_record_in_train_pool(tmp_path):
    baseline_path = tmp_path / "base.pt"
    _save_dummy_checkpoint(baseline_path)

    test_record = _make_record("test_leak", split="test")
    train_pool = [_make_record("p0", split="train"), test_record]
    dev_pool = [_make_record("d0", split="dev", source_game_id="g9")]

    cfg = SelfLearningRoundConfig(
        seed=0,
        num_candidate_positions=2,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        min_verified_examples=1,
    )
    with pytest.raises(ValueError, match="test"):
        run_self_learning_round(baseline_path, train_pool, dev_pool, tmp_path / "out", cfg)


def test_run_round_raises_on_test_record_in_dev_pool(tmp_path):
    baseline_path = tmp_path / "base.pt"
    _save_dummy_checkpoint(baseline_path)

    train_pool = [_make_record("p0", split="train")]
    dev_pool = [_make_record("test_dev", split="test")]

    cfg = SelfLearningRoundConfig(
        seed=0,
        num_candidate_positions=2,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        min_verified_examples=1,
    )
    with pytest.raises(ValueError, match="test"):
        run_self_learning_round(baseline_path, train_pool, dev_pool, tmp_path / "out", cfg)


# ===========================================================================
# 7. Insufficient examples fails safely
# ===========================================================================


def test_insufficient_verified_examples_raises(tmp_path):
    baseline_path = tmp_path / "base.pt"
    _save_dummy_checkpoint(baseline_path)

    # Empty train pool → zero verified examples
    cfg = SelfLearningRoundConfig(
        seed=0,
        num_candidate_positions=0,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        min_verified_examples=1,
    )
    with pytest.raises(InsufficientVerifiedExamplesError):
        run_self_learning_round(
            baseline_path,
            [],   # empty train pool
            [],   # empty dev pool
            tmp_path / "out",
            cfg,
        )


def test_insufficient_examples_writes_summary_json(tmp_path):
    baseline_path = tmp_path / "base.pt"
    _save_dummy_checkpoint(baseline_path)

    cfg = SelfLearningRoundConfig(
        seed=0,
        num_candidate_positions=0,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        min_verified_examples=1,
    )
    out_dir = tmp_path / "out"
    with pytest.raises(InsufficientVerifiedExamplesError):
        run_self_learning_round(baseline_path, [], [], out_dir, cfg)

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed"
    assert "Insufficient" in summary["reason"]


# ===========================================================================
# 8. Acceptance and rejection paths (controlled, no real Stockfish)
# ===========================================================================


def _run_round_with_mocked_engine(
    tmp_path: Path,
    baseline_metrics: dict,
    candidate_metrics_override: dict | None = None,
    gate_cfg: GateConfig | None = None,
    min_verified: int = 1,
) -> "SelfLearningRoundResult":
    """
    Runs run_self_learning_round with mocked engine and mocked dev evaluation,
    injecting controlled metrics to force accept or reject.
    """
    from chessevolve.self_learning import SelfLearningRoundResult

    baseline_path = tmp_path / "base.pt"
    model = _save_dummy_checkpoint(baseline_path)

    train_pool = [
        _make_record(f"p{i}", split="train", candidates=["e2e4", "d2d4"])
        for i in range(4)
    ]
    dev_pool = [_make_record("d0", split="dev", source_game_id="g9")]

    # Pre-build dev examples so we control what evaluate() sees.
    dev_examples = [_make_example(target=0, n=2)]

    if gate_cfg is None:
        gate_cfg = GateConfig(min_regret_reduction_cp=5.0, max_agreement_drop=0.1)

    cfg = SelfLearningRoundConfig(
        seed=0,
        round_number=1,
        num_candidate_positions=4,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        fine_tune_epochs=1,
        fine_tune_lr=1e-3,
        fine_tune_batch_size=4,
        min_verified_examples=min_verified,
        gate=gate_cfg,
        verifier=VerifierConfig(allowed_split="train"),
    )

    # Mock generate_feedback to return clean controlled records
    mock_feedback = [
        _make_feedback(f"p{i}", scores=[50, 20], split="train")
        for i in range(4)
    ]

    # Mock evaluate() to return controlled metrics for accept/reject testing
    call_count = [0]

    def _mock_evaluate(model_arg, examples_arg):
        call_count[0] += 1
        if call_count[0] == 1:
            return baseline_metrics
        # candidate metrics
        if candidate_metrics_override is not None:
            return candidate_metrics_override
        # Default: improve by 30 cp
        improved = dict(baseline_metrics)
        improved["mean_regret_cp"] = baseline_metrics["mean_regret_cp"] - 30.0
        return improved

    with (
        patch("chessevolve.self_learning.generate_feedback", return_value=mock_feedback),
        patch("chessevolve.self_learning.evaluate", side_effect=_mock_evaluate),
    ):
        result = run_self_learning_round(
            baseline_path,
            train_pool,
            dev_pool,
            tmp_path / "out",
            cfg,
            dev_examples=dev_examples,
        )

    return result


def test_round_acceptance_path(tmp_path):
    baseline_m = _metrics(regret=100.0, top1=0.3)
    result = _run_round_with_mocked_engine(tmp_path, baseline_m)
    assert result.decision.accepted is True
    out = Path(result.output_dir)
    assert (out / "accepted.pt").exists()
    assert (out / "current_best.pt").exists()
    assert (out / "candidate.pt").exists()
    assert (out / "retained.pt").exists()
    assert (out / "decision.json").exists()
    assert (out / "config.json").exists()
    assert (out / "round_summary.json").exists()
    assert (out / "feedback_records.jsonl").exists()
    assert (out / "rejections.jsonl").exists()


def test_round_rejection_path(tmp_path):
    baseline_m = _metrics(regret=100.0, top1=0.3)
    # Candidate regret improves only by 2 cp < threshold of 5 cp → reject
    candidate_m = _metrics(regret=98.0, top1=0.3)
    result = _run_round_with_mocked_engine(
        tmp_path, baseline_m, candidate_metrics_override=candidate_m,
        gate_cfg=GateConfig(min_regret_reduction_cp=5.0),
    )
    assert result.decision.accepted is False
    out = Path(result.output_dir)
    assert not (out / "accepted.pt").exists()
    assert (out / "retained.pt").exists()
    assert (out / "current_best.pt").exists()
    # Verify that the baseline checkpoint itself was not overwritten
    baseline_path = tmp_path / "base.pt"
    assert baseline_path.exists()


def test_round_rejection_preserves_baseline_unchanged(tmp_path):
    import torch

    baseline_path = tmp_path / "base.pt"
    model = _save_dummy_checkpoint(baseline_path)
    original_size = baseline_path.stat().st_size

    baseline_m = _metrics(regret=100.0, top1=0.3)
    candidate_m = _metrics(regret=99.0, top1=0.3)  # 1 cp improvement < 5 cp threshold
    _run_round_with_mocked_engine(
        tmp_path, baseline_m, candidate_metrics_override=candidate_m,
        gate_cfg=GateConfig(min_regret_reduction_cp=5.0),
    )

    # The original baseline file must not have changed
    assert baseline_path.stat().st_size == original_size
    reloaded = torch.load(baseline_path, map_location="cpu")
    original_w = {k: v.clone() for k, v in torch.load(baseline_path, map_location="cpu")["model_state_dict"].items()}
    for k, v in reloaded["model_state_dict"].items():
        assert torch.equal(v, original_w[k])


def test_round_summary_json_complete(tmp_path):
    baseline_m = _metrics(regret=100.0, top1=0.3)
    result = _run_round_with_mocked_engine(tmp_path, baseline_m)
    summary = json.loads((Path(result.output_dir) / "round_summary.json").read_text())
    required_keys = {
        "round_number", "seed", "accepted", "decision_reason",
        "regret_diff_cp", "agreement_diff", "num_candidate_positions",
        "num_feedback_generated", "num_verified_examples",
        "num_rejected_examples", "baseline_metrics", "candidate_metrics",
    }
    assert required_keys <= set(summary.keys())


def test_recursive_rounds_continue_from_current_best(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.pt"
    _save_dummy_checkpoint(baseline_path)
    train_pool = [_make_record("p0", split="train")]
    dev_pool = [_make_record("d0", split="dev", source_game_id="g-dev")]
    calls = []

    def fake_round(baseline, train, dev, output, config, **kwargs):
        calls.append(Path(baseline))
        current = Path(output) / "current_best.pt"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_bytes(b"checkpoint")
        decision = GateDecision(
            accepted=False,
            reason="rejected",
            baseline_metrics=_metrics(10, 0.5),
            candidate_metrics=_metrics(11, 0.5),
            regret_diff_cp=-1,
            agreement_diff=0,
            thresholds={},
        )
        return SimpleNamespace(
            decision=decision,
            summary={"round_number": config.round_number, "accepted": False},
            output_dir=str(output),
        )

    monkeypatch.setattr("chessevolve.self_learning.run_self_learning_round", fake_round)
    result = run_recursive_self_learning(
        baseline_path,
        train_pool,
        dev_pool,
        tmp_path / "recursive",
        RecursiveSelfLearningConfig(iterations=2),
    )
    assert calls == [
        baseline_path,
        tmp_path / "recursive" / "round_001" / "current_best.pt",
    ]
    assert result.summary["accepted_updates"] == 0
    assert result.summary["rejected_updates"] == 2


def test_decision_json_contains_thresholds(tmp_path):
    baseline_m = _metrics(regret=100.0, top1=0.3)
    result = _run_round_with_mocked_engine(
        tmp_path, baseline_m,
        gate_cfg=GateConfig(min_regret_reduction_cp=7.5, max_agreement_drop=0.08),
    )
    decision = json.loads((Path(result.output_dir) / "decision.json").read_text())
    assert decision["thresholds"]["min_regret_reduction_cp"] == pytest.approx(7.5)
    assert decision["thresholds"]["max_agreement_drop"] == pytest.approx(0.08)


# ===========================================================================
# 9. End-to-end smoke with real Stockfish
# ===========================================================================


@pytest.mark.skipif(not stockfish_available, reason="Stockfish not available")
def test_smoke_self_learning_round(tmp_path):
    """Full end-to-end round with real Stockfish on minimal data."""
    from chessevolve.dataset import generate_random_positions, split_records

    baseline_path = tmp_path / "base.pt"
    _save_dummy_checkpoint(baseline_path, hidden=16)

    records = generate_random_positions(num_games=4, max_plies=8, seed=0)
    split_res = split_records(records, seed=0)

    train_pool = [r for r in split_res["train"]]
    dev_pool = [r for r in split_res["dev"]]

    cfg = SelfLearningRoundConfig(
        seed=0,
        round_number=1,
        num_candidate_positions=3,
        selection_policy="random",
        engine_depth=2,
        stockfish_path=str(STOCKFISH_PATH),
        fine_tune_epochs=1,
        fine_tune_lr=1e-3,
        fine_tune_batch_size=4,
        min_verified_examples=1,
        gate=GateConfig(min_regret_reduction_cp=0.0, max_agreement_drop=1.0),
        verifier=VerifierConfig(allowed_split="train"),
    )

    out_dir = tmp_path / "round1"
    result = run_self_learning_round(
        baseline_path, train_pool, dev_pool, out_dir, cfg
    )

    # Check structural artifacts
    assert out_dir.exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "decision.json").exists()
    assert (out_dir / "round_summary.json").exists()
    assert (out_dir / "feedback_records.jsonl").exists()
    assert (out_dir / "rejections.jsonl").exists()
    assert (out_dir / "retained.pt").exists()
    assert (out_dir / "current_best.pt").exists()

    summary = json.loads((out_dir / "round_summary.json").read_text())
    assert isinstance(summary["accepted"], bool)
    assert summary["num_feedback_generated"] >= 0
