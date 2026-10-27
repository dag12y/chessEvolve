"""Tests for baseline training helpers."""
from __future__ import annotations

import torch
import math
from pathlib import Path

import pytest

from chessevolve.dataset import TrainingExample
from chessevolve.encoding import POSITION_DIM, MOVE_DIM
from chessevolve.train import (
    TrainConfig,
    make_targets,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    run_training,
    pairwise_ranking_loss,
)
from chessevolve.model import MLPRanker, ModelConfig


def _fake_example(target=2, n=5):
    import numpy as np
    scores = [10, 20, 50, 30, 40]
    return TrainingExample(
        position_features=np.zeros(POSITION_DIM, dtype=__import__("numpy").float32),
        move_features=__import__("numpy").zeros((n, MOVE_DIM), dtype=__import__("numpy").float32),
        candidate_scores=scores,
        target_index=target,
        position_id="p",
        fen="startpos",
    )


def test_make_targets_argmax():
    ex = _fake_example()
    assert make_targets(ex) == 2
    # mate score handling: large value wins
    ex2 = _fake_example()
    ex2.candidate_scores = [100, -100000, 50]
    ex2.move_features = __import__("numpy").zeros((3, MOVE_DIM), dtype=__import__("numpy").float32)
    assert make_targets(ex2) == 0


def test_pairwise_ranking_loss_is_finite_and_rewards_correct_order():
    good = torch.tensor([2.0, 0.0, -1.0], requires_grad=True)
    bad = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    scores = [300, 100, -100]
    good_loss = pairwise_ranking_loss(good, scores)
    bad_loss = pairwise_ranking_loss(bad, scores)
    assert torch.isfinite(good_loss)
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None


def test_mixed_loss_is_finite():
    examples = [_fake_example()]
    model = MLPRanker(ModelConfig(hidden_dim=8, num_hidden_layers=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(
        model,
        examples,
        optimizer,
        batch_size=1,
        loss_mode="mixed",
        classification_weight=0.5,
    )
    assert math.isfinite(loss)


def test_candidate_label_alignment():
    ex = _fake_example(target=1)
    assert ex.target_index == 1
    assert 0 <= ex.target_index < len(ex.candidate_scores)


def test_score_perspective_consistent():
    # Higher score = better for side to move; argmax picks best.
    ex = _fake_example()
    assert ex.candidate_scores[ex.target_index] == max(ex.candidate_scores)


def test_train_one_epoch_finite():
    model = MLPRanker(ModelConfig(hidden_dim=16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    examples = [_fake_example() for _ in range(4)]
    loss = train_one_epoch(model, examples, opt, batch_size=2)
    assert loss == loss  # not nan
    import math
    assert math.isfinite(loss)


def test_evaluate_finite():
    model = MLPRanker(ModelConfig(hidden_dim=16))
    examples = [_fake_example() for _ in range(3)]
    m = evaluate(model, examples)
    assert "top1_agreement" in m
    assert "mean_regret_cp" in m
    assert m["num_examples"] == 3


def test_checkpoint_reload(tmp_path):
    model = MLPRanker(ModelConfig(hidden_dim=16))
    cfg = TrainConfig(epochs=1, num_games=2, max_plies=4, engine_depth=2)
    path = tmp_path / "c.pt"
    save_checkpoint(path, model, cfg, {"n": 1}, {"loss": 1.0})
    m2, ckpt = load_checkpoint(path)
    assert m2.config.hidden_dim == 16
    assert ckpt["train_config"]["epochs"] == 1


stockfish_available = Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal").exists()


@pytest.mark.skipif(not stockfish_available, reason="stockfish not available")
def test_smoke_training_run(tmp_path):
    cfg = TrainConfig(seed=1, epochs=1, num_games=3, max_plies=6, engine_depth=2,
                      stockfish_path="tools/stockfish/stockfish/stockfish-linux-x86-64-universal")
    out = tmp_path / "run"
    result = run_training(cfg, out)
    assert (out / "best.pt").exists()
    assert (out / "final.pt").exists()
    assert len(result["history"]) == 1
    # checkpoint reload works
    model, ckpt = load_checkpoint(out / "best.pt")
    assert model.parameter_count() > 0
    # verify evaluated datasets and manifest are persisted
    assert (out / "datasets" / "train_records.jsonl").exists()
    assert (out / "datasets" / "dev_records.jsonl").exists()
    assert (out / "datasets" / "manifest.json").exists()


def test_gradient_accumulation_scaling_full_and_partial():
    """Deterministic comparison demonstrating full and partial batches scale gradients correctly."""
    import copy
    import numpy as np
    import torch.nn.functional as F

    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    examples = []
    for i in range(5):
        ex = TrainingExample(
            position_features=rng.randn(POSITION_DIM).astype(np.float32),
            move_features=rng.randn(4, MOVE_DIM).astype(np.float32),
            candidate_scores=[10, 20, 30, 40],
            target_index=i % 4,
            position_id=f"p_{i}",
            fen="startpos",
        )
        examples.append(ex)

    model = MLPRanker(ModelConfig(hidden_dim=32, num_hidden_layers=1))

    # Analytical reference for Batch 1 (full batch of 3, scaled by 1/3)
    m_ref1 = copy.deepcopy(model)
    m_ref1.train()
    loss_b1 = 0
    for ex in examples[:3]:
        p = torch.from_numpy(ex.position_features).float()
        m = torch.from_numpy(ex.move_features).float()
        t = torch.tensor(ex.target_index, dtype=torch.long)
        loss_b1 = loss_b1 + F.cross_entropy(m_ref1(p, m).unsqueeze(0), t.unsqueeze(0))
    (loss_b1 / 3.0).backward()
    ref_grads1 = [p.grad.clone() for p in m_ref1.parameters()]

    # Analytical reference for Batch 2 (partial batch of 2, scaled by 1/2)
    m_ref2 = copy.deepcopy(model)
    m_ref2.train()
    loss_b2 = 0
    for ex in examples[3:]:
        p = torch.from_numpy(ex.position_features).float()
        m = torch.from_numpy(ex.move_features).float()
        t = torch.tensor(ex.target_index, dtype=torch.long)
        loss_b2 = loss_b2 + F.cross_entropy(m_ref2(p, m).unsqueeze(0), t.unsqueeze(0))
    (loss_b2 / 2.0).backward()
    ref_grads2 = [p.grad.clone() for p in m_ref2.parameters()]

    class RecordingOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, defaults={})
            self.recorded = []
        def step(self, closure=None):
            self.recorded.append([p.grad.clone() for p in self.param_groups[0]["params"]])

    test_model = copy.deepcopy(model)
    rec_opt = RecordingOptimizer(test_model.parameters())
    train_one_epoch(test_model, examples, rec_opt, batch_size=3)

    assert len(rec_opt.recorded) == 2
    # Verify Batch 1 gradients match ref_grads1 (1/3)
    for g_actual, g_ref in zip(rec_opt.recorded[0], ref_grads1):
        assert torch.allclose(g_actual, g_ref, atol=1e-6)
    # Verify Batch 2 gradients match ref_grads2 (1/2, not 1/3)
    for g_actual, g_ref in zip(rec_opt.recorded[1], ref_grads2):
        assert torch.allclose(g_actual, g_ref, atol=1e-6)
    # Verify that Batch 2 was NOT scaled blindly by 3
    for g_actual, g_ref in zip(rec_opt.recorded[1], ref_grads2):
        if torch.abs(g_actual).max() > 1e-5:
            assert not torch.allclose(g_actual, g_ref * (2.0 / 3.0), atol=1e-4)


def test_train_one_epoch_mean_loss_objective():
    """Preserve mean-loss objective across different batch sizes."""
    import numpy as np

    torch.manual_seed(99)
    rng = np.random.RandomState(99)
    examples = []
    for i in range(5):
        ex = TrainingExample(
            position_features=rng.randn(POSITION_DIM).astype(np.float32),
            move_features=rng.randn(3, MOVE_DIM).astype(np.float32),
            candidate_scores=[10, 20, 30],
            target_index=i % 3,
            position_id=f"p_{i}",
            fen="startpos",
        )
        examples.append(ex)

    model = MLPRanker(ModelConfig(hidden_dim=16, dropout=0.0))
    model.eval()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    loss_b1 = train_one_epoch(model, examples, opt, batch_size=1)
    loss_b2 = train_one_epoch(model, examples, opt, batch_size=2)
    loss_b5 = train_one_epoch(model, examples, opt, batch_size=5)

    assert abs(loss_b1 - loss_b2) < 1e-6
    assert abs(loss_b1 - loss_b5) < 1e-6


def test_train_one_epoch_invalid_batch_size():
    model = MLPRanker(ModelConfig(hidden_dim=16))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        train_one_epoch(model, [_fake_example()], opt, batch_size=0)


def test_shuffle_reproducibility():
    """Seeded shuffle reproduces the same training run, but epochs differ."""
    import random
    examples = [_fake_example(target=i % 4) for i in range(10)]
    for i, ex in enumerate(examples):
        ex.position_id = f"pos_{i}"

    rng1 = random.Random(42)
    # Epoch 1
    ex_e1 = list(examples)
    rng1.shuffle(ex_e1)
    order_run1_epoch1 = [e.position_id for e in ex_e1]

    # Epoch 2
    ex_e2 = list(examples)
    rng1.shuffle(ex_e2)
    order_run1_epoch2 = [e.position_id for e in ex_e2]

    # Simulate run 2 with same seed
    rng2 = random.Random(42)
    ex2_e1 = list(examples)
    rng2.shuffle(ex2_e1)
    order_run2_epoch1 = [e.position_id for e in ex2_e1]

    ex2_e2 = list(examples)
    rng2.shuffle(ex2_e2)
    order_run2_epoch2 = [e.position_id for e in ex2_e2]

    assert order_run1_epoch1 == order_run2_epoch1
    assert order_run1_epoch2 == order_run2_epoch2
    assert order_run1_epoch1 != order_run1_epoch2
    assert [e.position_id for e in examples] == [f"pos_{i}" for i in range(10)]


def test_tiny_batch_overfit():
    """Automated sanity check: MLP ranker must overfit a tiny fixed synthetic batch."""
    import numpy as np

    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    examples = []
    for i in range(4):
        n_cands = 4
        target = i % n_cands
        scores = [10] * n_cands
        scores[target] = 100
        ex = TrainingExample(
            position_features=rng.randn(POSITION_DIM).astype(np.float32),
            move_features=rng.randn(n_cands, MOVE_DIM).astype(np.float32),
            candidate_scores=scores,
            target_index=target,
            position_id=f"syn_{i}",
            fen="syn_fen",
        )
        examples.append(ex)

    model = MLPRanker(ModelConfig(hidden_dim=64, num_hidden_layers=2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    init_metrics = evaluate(model, examples)
    assert init_metrics["loss"] > 0.5

    for _ in range(40):
        train_one_epoch(model, examples, opt, batch_size=4)

    final_metrics = evaluate(model, examples)

    assert final_metrics["loss"] < 0.2
    assert final_metrics["loss"] < init_metrics["loss"] * 0.2
    assert final_metrics["top1_agreement"] == 1.0
    assert final_metrics["mean_regret_cp"] == 0.0
