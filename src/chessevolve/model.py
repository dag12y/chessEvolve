"""Small PyTorch MLP ranker for chess candidate moves."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

from .encoding import (
    DEFAULT_FEATURE_GROUP,
    FEATURE_GROUPS,
    MOVE_DIM,
    POSITION_DIM,
    LEGACY_MOVE_DIM,
    LEGACY_POSITION_DIM,
    POSITION_FEATURES_C,
)


@dataclass
class ModelConfig:
    feature_group: str = DEFAULT_FEATURE_GROUP
    position_dim: int = POSITION_DIM
    move_dim: int = MOVE_DIM
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_feature_group(cls, feature_group: str, **kwargs: object) -> "ModelConfig":
        if feature_group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {feature_group!r}")
        position_dim = {
            "A": LEGACY_POSITION_DIM,
            "B": LEGACY_POSITION_DIM,
            "C": POSITION_FEATURES_C,
            "D": POSITION_DIM,
        }[feature_group]
        move_dim = LEGACY_MOVE_DIM if feature_group == "A" else MOVE_DIM
        return cls(
            feature_group=feature_group,
            position_dim=position_dim,
            move_dim=move_dim,
            **kwargs,
        )


class MLPRanker(nn.Module):
    """Ranker that scores each candidate move for a single position.

    Forward input:
      position: (position_dim,) or (batch, position_dim)  -- one position
      moves:    (num_candidates, move_dim) or (batch, num_candidates, move_dim)
    Output: (num_candidates,) or (batch, num_candidates) scalar scores.

    Candidates from different positions are never mixed: the caller passes one
    position's candidates at a time (or a batch where each row is one position).
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        input_dim = self.config.position_dim + self.config.move_dim
        layers = []
        in_dim = input_dim
        for _ in range(self.config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            layers.append(nn.ReLU())
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = self.config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, position: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        # Normalize shapes.
        if position.dim() == 1:
            position = position.unsqueeze(0)  # (1, P)
        if moves.dim() == 2:
            moves = moves.unsqueeze(0)  # (1, N, M)
        # position: (B, P), moves: (B, N, M)
        b, n, m = moves.shape
        pos_expanded = position.unsqueeze(1).expand(b, n, -1)  # (B, N, P)
        combined = torch.cat([pos_expanded, moves], dim=-1)  # (B, N, P+M)
        flat = combined.reshape(b * n, -1)
        out = self.net(flat).reshape(b, n)  # (B, N)
        if b == 1:
            out = out.squeeze(0)  # (N,)
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
