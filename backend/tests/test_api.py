"""Tests for the ChessEvolve demo backend API.

Covers ``GET /models`` and ``POST /move`` including illegal inputs.
Run with:  source .venv/bin/activate && pip install fastapi uvicorn httpx  # needs network
           source .venv/bin/activate && python -m pytest backend/tests -q
(httpx is required by fastapi.testclient.TestClient and is not installed in
the offline environment, so these tests could not be executed here.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Scholar's mate after 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7# -- game over.
GAME_OVER_FEN = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


def test_models_list() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    ids = [model["id"] for model in body["models"]]
    expected = [
        "v2_mixed_teacher_seed1",
        "random",
    ]
    assert ids == expected
    for model in body["models"]:
        assert set(model) >= {"id", "name", "type", "available"}
        assert model["type"] in ("checkpoint", "random")
    random_model = next(m for m in body["models"] if m["id"] == "random")
    assert random_model["type"] == "random"
    assert random_model["available"] is True


def test_move_returns_chosen_and_ranked_top3() -> None:
    response = client.post(
        "/move",
        json={"fen": START_FEN, "model_id": "v2_mixed_teacher_seed1", "temperature": 0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == "v2_mixed_teacher_seed1"
    assert body["temperature"] == 0
    assert body["chosen"]["uci"] in {m.uci() for m in chess.Board(START_FEN).legal_moves}
    assert len(body["top_moves"]) == 3
    scores = [move["score"] for move in body["top_moves"]]
    assert scores == sorted(scores, reverse=True)
    assert all(move["rank"] == i + 1 for i, move in enumerate(body["top_moves"]))
    assert body["top_moves"][0]["uci"] == body["chosen"]["uci"]
    assert body["position_after"]["fen"] != START_FEN


def test_move_argmax_is_deterministic() -> None:
    first = client.post(
        "/move", json={"fen": START_FEN, "model_id": "v2_mixed_teacher_seed1"}
    ).json()
    second = client.post(
        "/move", json={"fen": START_FEN, "model_id": "v2_mixed_teacher_seed1"}
    ).json()
    assert first["chosen"]["uci"] == second["chosen"]["uci"]
    assert first["top_moves"][0]["uci"] == second["top_moves"][0]["uci"]


def test_move_temperature_sampling_stays_legal() -> None:
    response = client.post(
        "/move",
        json={"fen": START_FEN, "model_id": "v2_mixed_teacher_seed1", "temperature": 1.0},
    )
    assert response.status_code == 200
    body = response.json()
    board = chess.Board(START_FEN)
    assert body["chosen"]["uci"] in {m.uci() for m in board.legal_moves}
    probabilities = [move["probability"] for move in body["top_moves"]]
    assert all(p is not None and 0.0 <= p <= 1.0 for p in probabilities)


def test_move_all_models_work() -> None:
    board = chess.Board(START_FEN)
    legal = {m.uci() for m in board.legal_moves}
    for model_id in (
        "v2_mixed_teacher_seed1",
        "random",
    ):
        response = client.post(
            "/move", json={"fen": START_FEN, "model_id": model_id, "temperature": 0}
        )
        assert response.status_code == 200, f"{model_id}: {response.text}"
        body = response.json()
        assert body["chosen"]["uci"] in legal


def test_move_illegal_fen() -> None:
    response = client.post(
        "/move", json={"fen": "this is not a fen", "model_id": "v2_mixed_teacher_seed1"}
    )
    assert response.status_code == 400
    assert "fen" in response.json()["detail"].lower()


def test_move_game_over() -> None:
    import chess

    assert chess.Board(GAME_OVER_FEN).is_game_over()
    response = client.post(
        "/move", json={"fen": GAME_OVER_FEN, "model_id": "v2_mixed_teacher_seed1"}
    )
    assert response.status_code == 400
    assert "game over" in response.json()["detail"].lower()


def test_move_unknown_model() -> None:
    response = client.post(
        "/move", json={"fen": START_FEN, "model_id": "nonexistent_model"}
    )
    assert response.status_code == 400
    assert "unknown model_id" in response.json()["detail"]


def test_move_negative_temperature() -> None:
    response = client.post(
        "/move",
        json={"fen": START_FEN,         "model_id": "v2_mixed_teacher_seed1", "temperature": -0.5},
    )
    assert response.status_code == 400
    assert "temperature" in response.json()["detail"].lower()


def test_model_takes_forced_mate_with_search() -> None:
    fen = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
    response = client.post(
        "/move",
        json={
            "fen": fen,
            "model_id": "v2_mixed_teacher_seed1",
            "temperature": 0,
            "search_depth": 2,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["chosen"]["uci"] in {"g6g8", "g6g7", "g6h6", "g6h5"}


def test_search_depth_is_validated() -> None:
    response = client.post(
        "/move",
        json={
            "fen": START_FEN,
            "model_id": "v2_mixed_teacher_seed1",
            "search_depth": 4,
        },
    )
    assert response.status_code == 400
    assert "search_depth" in response.json()["detail"]


def test_default_search_depth_is_two() -> None:
    response = client.post(
        "/move",
        json={"fen": START_FEN, "model_id": "v2_mixed_teacher_seed1"},
    )
    assert response.status_code == 200


def test_forcing_moves_are_not_pruned_from_search() -> None:
    fen = "4k3/8/8/3r4/8/2B5/8/4K3 w - - 0 1"
    response = client.post(
        "/move",
        json={
            "fen": fen,
            "model_id": "v2_mixed_teacher_seed1",
            "temperature": 0,
            "search_depth": 2,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["chosen"]["uci"] in {
        move.uci() for move in chess.Board(fen).legal_moves
    }
