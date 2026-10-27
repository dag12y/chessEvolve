"""ChessEvolve demo web backend (CPU only).

FastAPI service that ranks legal candidate moves for a FEN using the
ChessEvolve MLPRanker checkpoints. Reuses the project's own encoding and
training helpers (``chessevolve.encoding``, ``chessevolve.train``) -- no
inference logic is reimplemented here.

Endpoints:
    GET  /models          -- list configured models (checkpoints + random)
    POST /move            -- pick a move for a FEN; returns the chosen UCI
                             move plus the top 3 ranked legal moves with scores
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import chess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chessevolve.encoding import (
    LEGACY_POSITION_DIM,
    encode_move,
    encode_move_legacy,
    encode_position,
    encode_position_legacy,
)
from chessevolve.train import load_checkpoint

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "backend" / "models.json"
DEFAULT_TEMPERATURE = 0.0
MAX_TOP_MOVES = 3

# CORS: the Vite dev server runs on http://localhost:5173 by default.
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def load_model_config() -> list[dict[str, Any]]:
    """Load the model registry from ``backend/models.json``."""
    data = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    return data["models"]


MODEL_CONFIG = load_model_config()
MODEL_BY_ID: dict[str, dict[str, Any]] = {entry["id"]: entry for entry in MODEL_CONFIG}

# --------------------------------------------------------------------------- #
# Model service (reuses chessevolve.encoding + chessevolve.train)
# --------------------------------------------------------------------------- #


class ModelService:
    """Loads checkpoints lazily and ranks legal candidate moves for a position.

    Checkpoint loading and scoring reuse ``chessevolve.train.load_checkpoint``
    and ``chessevolve.encoding`` -- the exact same code path used in training.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, dict[str, Any]] | None] = {}

    def _load(self, model_id: str) -> Any:
        entry = MODEL_BY_ID[model_id]
        if model_id in self._cache:
            return self._cache[model_id]
        if entry["type"] == "random":
            self._cache[model_id] = None
            return None
        checkpoint_path = (REPO_ROOT / entry["path"]).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        model, _ckpt = load_checkpoint(checkpoint_path)
        model.eval()
        self._cache[model_id] = (model, entry)
        return self._cache[model_id]

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
        """Numerically stable softmax over scaled logits."""
        scaled = (scores - scores.max()) / max(temperature, 1e-6)
        scaled = np.clip(scaled, -700, 700)
        exp = np.exp(scaled)
        total = exp.sum()
        if total <= 0 or not np.isfinite(total):
            return np.full_like(scores, 1.0 / len(scores))
        return exp / total

    def rank(
        self, board: chess.Board, model_id: str, temperature: float
    ) -> dict[str, Any]:
        """Return the chosen move and the top ranked legal moves for ``board``."""
        if temperature < 0:
            raise HTTPException(
                status_code=400,
                detail=f"temperature must be >= 0, got {temperature}",
            )
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise HTTPException(
                status_code=400,
                detail="no legal moves: the game is already over",
            )

        # --- Random opponent: no model involved --------------------------- #
        if model_id == "random":
            chosen = _random_move(legal_moves)
            top = [
                {"rank": i + 1, "uci": m.uci(), "score": None, "probability": None}
                for i, m in enumerate(legal_moves[:MAX_TOP_MOVES])
            ]
            return _build_response(
                board, chosen, None, temperature, top, model_id, len(legal_moves)
            )

        # --- Model-backed ranking ---------------------------------------- #
        loaded = self._load(model_id)
        if loaded is None:
            raise HTTPException(
                status_code=400, detail=f"model {model_id!r} is not loadable"
            )
        model, _entry = loaded

        position_encoder = (
            encode_position_legacy
            if model.config.position_dim == LEGACY_POSITION_DIM
            else encode_position
        )
        move_encoder = (
            encode_move_legacy if model.config.move_dim == 158 else encode_move
        )
        position_vec = position_encoder(board)
        move_vecs = np.stack([move_encoder(board, move) for move in legal_moves])
        position_tensor = torch.tensor(position_vec, dtype=torch.float32)
        moves_tensor = torch.tensor(move_vecs, dtype=torch.float32)

        with torch.no_grad():
            scores_tensor = model(position_tensor, moves_tensor)
        scores = scores_tensor.detach().cpu().numpy().astype(np.float64)

        order = np.argsort(scores)[::-1]
        probabilities = self._softmax(scores, temperature if temperature > 0 else 1.0)

        if temperature == 0:
            chosen_index = int(order[0])
        else:
            chosen_index = int(np.random.choice(len(legal_moves), p=probabilities))

        chosen = legal_moves[chosen_index]
        top_moves = []
        for rank_idx, move_index in enumerate(order[:MAX_TOP_MOVES], start=1):
            move = legal_moves[move_index]
            top_moves.append(
                {
                    "rank": rank_idx,
                    "uci": move.uci(),
                    "score": float(scores[move_index]),
                    "probability": float(probabilities[move_index]),
                }
            )

        return _build_response(
            board,
            chosen,
            float(scores[chosen_index]),
            temperature,
            top_moves,
            model_id,
            len(legal_moves),
        )


def _random_move(legal_moves: list[chess.Move]) -> chess.Move:
    return legal_moves[int(np.random.randint(len(legal_moves)))]


def _build_response(
    board: chess.Board,
    chosen: chess.Move,
    chosen_score: float | None,
    temperature: float,
    top_moves: list[dict[str, Any]],
    model_id: str,
    num_candidates: int,
) -> dict[str, Any]:
    """Apply the chosen move and build the JSON response."""
    san = board.san(chosen)
    board.push(chosen)
    return {
        "model_id": model_id,
        "fen": board.fen(),
        "temperature": float(temperature),
        "chosen": {
            "uci": chosen.uci(),
            "san": san,
            "score": chosen_score,
        },
        "top_moves": top_moves,
        "legal_moves": num_candidates,
        "position_after": {
            "fen": board.fen(),
            "check": board.is_check(),
            "game_over": board.is_game_over(),
            "result": board.result(claim_draw=True),
        },
    }


# --------------------------------------------------------------------------- #
# Request/Response schemas
# --------------------------------------------------------------------------- #


class MoveRequest(BaseModel):
    fen: str
    model_id: str
    temperature: float | None = DEFAULT_TEMPERATURE


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

app = FastAPI(title="ChessEvolve Demo Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService()


@app.get("/models")
def list_models() -> dict[str, Any]:
    """List the configured models with availability status."""
    models_out: list[dict[str, Any]] = []
    for entry in MODEL_CONFIG:
        if entry["type"] == "checkpoint":
            path = (REPO_ROOT / entry["path"]).resolve()
            models_out.append(
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "type": "checkpoint",
                    "path": str(path),
                    "available": path.exists(),
                }
            )
        else:
            models_out.append(
                {"id": entry["id"], "name": entry["name"], "type": "random", "available": True}
            )
    return {"models": models_out}


@app.post("/move")
def get_move(request: MoveRequest) -> dict[str, Any]:
    """Choose a move for a FEN and rank the top legal candidates."""
    if request.model_id not in MODEL_BY_ID:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model_id {request.model_id!r}. "
            f"Known ids: {sorted(MODEL_BY_ID)}",
        )

    # Validate the FEN and reject finished games up front.
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"illegal FEN: {exc}") from exc

    if board.is_game_over():
        raise HTTPException(
            status_code=400,
            detail=f"game over ({board.result(claim_draw=True)}): no move can be played",
        )

    temperature = (
        request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    )
    try:
        return service.rank(board, request.model_id, temperature)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
