"""Tests for the Chess.com PGN downloader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from chessevolve.downloader import (
    API_BASE_URL,
    DownloadError,
    _retry_after_seconds,
    build_archives_url,
    confirm_overwrite,
    download_player_games,
    extract_game_pdns,
    fetch_json,
    join_games,
    month_label,
    output_path_for,
    parse_archives,
    save_pgn,
)


def test_build_archives_url_and_month_label():
    assert (
        build_archives_url("hikaru")
        == f"{API_BASE_URL}/player/hikaru/games/archives"
    )
    assert (
        month_label("https://api.chess.com/pub/player/hikaru/games/2026/05")
        == "2026/05"
    )
    assert month_label("invalid") == "invalid"


def test_parse_archives_valid():
    payload = {
        "archives": [
            "https://api.chess.com/pub/player/test/games/2026/01",
            "https://api.chess.com/pub/player/test/games/2026/02",
        ]
    }
    urls = parse_archives(payload)
    assert len(urls) == 2
    assert urls[0].endswith("2026/01")


def test_parse_archives_invalid():
    with pytest.raises(DownloadError, match="Expected an archives JSON object"):
        parse_archives(["not a dict"])

    with pytest.raises(DownloadError, match="contains no monthly archives"):
        parse_archives({"archives": []})

    with pytest.raises(DownloadError, match="Unexpected archive URL"):
        parse_archives({"archives": ["invalid_url"]})


def test_extract_game_pdns_and_join():
    payload = {
        "games": [
            {"pgn": '1. e4 e5 2. Nf3\n\n'},
            {"pgn": '1. d4 d5\n'},
            {"other": "data"},
        ]
    }
    pgns = extract_game_pdns(payload)
    assert len(pgns) == 2
    assert pgns[0] == "1. e4 e5 2. Nf3"
    assert pgns[1] == "1. d4 d5"

    assert extract_game_pdns([]) == []
    assert extract_game_pdns({"games": "not a list"}) == []

    joined = join_games([pgns, ["1. c4 c5"]])
    assert "1. e4 e5 2. Nf3\n\n1. d4 d5\n\n1. c4 c5" == joined


def test_retry_after_seconds():
    response = MagicMock()
    response.headers = {"Retry-After": "5.5"}
    assert _retry_after_seconds(response) == 5.5

    response.headers = {"Retry-After": "invalid"}
    assert _retry_after_seconds(response) is None

    response.headers = None
    assert _retry_after_seconds(response) is None


def test_fetch_json_success():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"archives": ["https://foo"]}
    session.get.return_value = response

    res = fetch_json(session, "https://test.url", sleep=lambda _: None)
    assert res == {"archives": ["https://foo"]}


def test_fetch_json_404():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    session.get.return_value = response

    with pytest.raises(DownloadError, match="Not found \\(HTTP 404\\)"):
        fetch_json(session, "https://test.url", sleep=lambda _: None)


def test_fetch_json_retries_on_rate_limit():
    session = MagicMock()
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"success": True}

    session.get.side_effect = [resp_429, resp_200]
    sleeps = []
    res = fetch_json(session, "https://test.url", sleep=sleeps.append)
    assert res == {"success": True}
    assert len(sleeps) == 1
    assert sleeps[0] == 0.01


def test_fetch_json_max_retries_exceeded():
    session = MagicMock()
    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.headers = {}
    session.get.return_value = resp_500

    with pytest.raises(DownloadError, match="HTTP 500 after 3 attempts"):
        fetch_json(session, "https://test.url", sleep=lambda _: None)


def test_fetch_json_invalid_json():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("bad json")
    session.get.return_value = response

    with pytest.raises(DownloadError, match="Invalid JSON response"):
        fetch_json(session, "https://test.url", sleep=lambda _: None)


def test_download_player_games_full_flow():
    session = MagicMock()
    # 1. Archives response
    resp_archives = MagicMock()
    resp_archives.status_code = 200
    resp_archives.json.return_value = {
        "archives": [
            "https://api.chess.com/pub/player/hikaru/games/2026/01",
            "https://api.chess.com/pub/player/hikaru/games/2026/02",
        ]
    }

    # 2. Monthly game payloads
    resp_month1 = MagicMock()
    resp_month1.status_code = 200
    resp_month1.json.return_value = {
        "games": [{"pgn": "1. e4 e5"}, {"pgn": "1. d4 d5"}]
    }

    resp_month2 = MagicMock()
    resp_month2.status_code = 200
    resp_month2.json.return_value = {"games": [{"pgn": "1. c4 e5"}]}

    session.get.side_effect = [resp_archives, resp_month1, resp_month2]

    res = download_player_games(
        "hikaru",
        session=session,
        sleep=lambda _: None,
        delay=0.0,
        progress=lambda _: None,
    )
    assert res.username == "hikaru"
    assert res.archives_total == 2
    assert res.archives_processed == 2
    assert res.games_downloaded == 3
    assert res.failed_requests == []
    assert "1. e4 e5" in res.pgn
    assert "1. c4 e5" in res.pgn


def test_download_player_games_partial_failure():
    session = MagicMock()
    resp_archives = MagicMock()
    resp_archives.status_code = 200
    resp_archives.json.return_value = {
        "archives": [
            "https://api.chess.com/pub/player/hikaru/games/2026/01",
            "https://api.chess.com/pub/player/hikaru/games/2026/02",
        ]
    }

    resp_month1 = MagicMock()
    resp_month1.status_code = 200
    resp_month1.json.return_value = {"games": [{"pgn": "1. e4 e5"}]}

    # Month 2 returns 404
    resp_month2 = MagicMock()
    resp_month2.status_code = 404

    session.get.side_effect = [resp_archives, resp_month1, resp_month2]

    res = download_player_games(
        "hikaru",
        session=session,
        sleep=lambda _: None,
        delay=0.0,
        progress=lambda _: None,
    )
    assert res.archives_total == 2
    assert res.archives_processed == 1
    assert res.games_downloaded == 1
    assert len(res.failed_requests) == 1


def test_confirm_overwrite_and_save(tmp_path: Path):
    path = tmp_path / "test.pgn"
    assert confirm_overwrite(path) is True

    save_pgn("1. e4 e5", path)
    assert path.is_file()
    assert path.read_text().strip() == "1. e4 e5"

    assert confirm_overwrite(path, input_fn=lambda _: "y") is True
    assert confirm_overwrite(path, input_fn=lambda _: "n") is False
    assert confirm_overwrite(path, force=True) is True


def test_output_path_for():
    custom_dir = Path("my/data")
    assert output_path_for("magnus", custom_dir) == Path("my/data/magnus.pgn")
