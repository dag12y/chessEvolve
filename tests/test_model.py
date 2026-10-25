"""Tests for the MLP ranker."""
from __future__ import annotations

import torch

from chessevolve.encoding import POSITION_DIM, MOVE_DIM
from chessevolve.model import MLPRanker, ModelConfig


def test_output_shape_single_position():
    model = MLPRanker()
    pos = torch.zeros(POSITION_DIM)
    moves = torch.zeros(5, MOVE_DIM)
    out = model(pos, moves)
    assert out.shape == (5,)


def test_output_shape_batched():
    model = MLPRanker()
    pos = torch.zeros(2, POSITION_DIM)
    moves = torch.zeros(2, 4, MOVE_DIM)
    out = model(pos, moves)
    assert out.shape == (2, 4)


def test_varying_candidate_counts():
    model = MLPRanker()
    model.eval()
    for n in [1, 3, 20, 64]:
        pos = torch.randn(POSITION_DIM)
        moves = torch.randn(n, MOVE_DIM)
        out = model(pos, moves)
        assert out.shape == (n,)
        assert torch.isfinite(out).all()


def test_finite_loss_and_gradients():
    model = MLPRanker()
    pos = torch.randn(POSITION_DIM)
    moves = torch.randn(6, MOVE_DIM)
    target = torch.tensor(2, dtype=torch.long)
    logits = model(pos, moves)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_checkpoint_save_load(tmp_path):
    model = MLPRanker(ModelConfig(hidden_dim=32))
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config.to_dict()}, path)
    ckpt = torch.load(path, map_location="cpu")
    m2 = MLPRanker(ModelConfig(**ckpt["model_config"]))
    m2.load_state_dict(ckpt["model_state_dict"])
    # outputs match
    pos = torch.randn(POSITION_DIM)
    moves = torch.randn(4, MOVE_DIM)
    model.eval(); m2.eval()
    with torch.no_grad():
        a = model(pos, moves)
        b = m2(pos, moves)
    assert torch.allclose(a, b)


def test_parameter_count_small():
    model = MLPRanker()
    assert model.parameter_count() < 500_000
    assert model.parameter_count() > 0
