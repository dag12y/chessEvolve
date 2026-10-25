"""Downloader for player game archives from the Chess.com Published Data API.

Fetches monthly PGN archives over HTTPS sequentially without authentication,
following polite rate-limiting, timeout, retry, and User-Agent requirements.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

API_BASE_URL = "https://api.chess.com/pub"
USER_AGENT = (
    "ChessEvolve/0.1 "
    "(personal chess imitation learning project; educational research)"
)
REQUEST_TIMEOUT = (10.0, 30.0)
REQUEST_DELAY_SECONDS = 1.1
MAX_RETRIES = 3
DEFAULT_DATA_DIR = Path("data") / "raw"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class DownloadError(Exception):
    """Raised when a Chess.com API request cannot be completed."""


@dataclass
class DownloadResult:
    username: str
    archives_total: int = 0
    archives_processed: int = 0
    games_downloaded: int = 0
    failed_requests: list[str] = field(default_factory=list)
    pgn: str = ""


def build_archives_url(username: str) -> str:
    """Build the Chess.com API monthly archives endpoint URL for a player."""
    return f"{API_BASE_URL}/player/{username}/games/archives"


def month_label(archive_url: str) -> str:
    """Extract YYYY/MM label from an archive URL."""
    parts = archive_url.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else archive_url


def parse_archives(payload: Any) -> list[str]:
    """Extract list of monthly archive URLs from API payload."""
    if not isinstance(payload, dict):
        raise DownloadError("Expected an archives JSON object")
    archives = payload.get("archives")
    if not isinstance(archives, list) or not archives:
        raise DownloadError("API response contains no monthly archives")
    urls = []
    for entry in archives:
        if not isinstance(entry, str) or not entry.startswith("http"):
            raise DownloadError(f"Unexpected archive URL: {entry!r}")
        urls.append(entry)
    return urls


def extract_game_pdns(payload: Any) -> list[str]:
    """Extract list of PGN strings from a monthly archive JSON payload."""
    if not isinstance(payload, dict):
        return []
    games = payload.get("games")
    if not isinstance(games, list):
        return []
    return [
        game["pgn"].strip()
        for game in games
        if isinstance(game, dict)
        and isinstance(game.get("pgn"), str)
        and game["pgn"].strip()
    ]


# Alias for spelling consistency
extract_game_pgns = extract_game_pdns


def join_games(monthly_pdns: list[list[str]]) -> str:
    """Join nested list of PGN strings separated by double newlines."""
    return "\n\n".join(pgn for month in monthly_pdns for pgn in month)


def _retry_after_seconds(response: Any) -> float | None:
    """Extract Retry-After header as float seconds if available."""
    headers = getattr(response, "headers", None)
    if not isinstance(headers, dict):
        return None
    try:
        raw = headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def fetch_json(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[float, float] = REQUEST_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Fetch JSON with retries, timeout, and backoff on 429/5xx status codes."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError(f"Network error requesting {url}: {exc}") from exc
            sleep(2**attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise DownloadError(f"Invalid JSON response from {url}") from exc

        if response.status_code == 404:
            raise DownloadError(f"Not found (HTTP 404): {url}")

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                raise DownloadError(
                    f"HTTP {response.status_code} after {MAX_RETRIES} attempts: {url}"
                )
            wait = _retry_after_seconds(response)
            sleep(wait if wait is not None else 2**attempt)
            continue

        raise DownloadError(f"HTTP {response.status_code}: {url}")

    raise DownloadError(f"Failed to fetch {url}")


def download_player_games(
    username: str,
    *,
    session: requests.Session,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = REQUEST_DELAY_SECONDS,
    progress: Callable[[str], None] = print,
    max_archives: int | None = None,
) -> DownloadResult:
    """Download games across all monthly archives for a given player."""
    archives = parse_archives(
        fetch_json(session, build_archives_url(username), sleep=sleep)
    )
    if max_archives is not None and max_archives > 0:
        # Take the most recent archives first if bounded
        archives = archives[-max_archives:]

    monthly_games: list[list[str]] = []
    failures: list[str] = []

    for index, url in enumerate(archives, start=1):
        progress(f"[{index}/{len(archives)}] Downloading {month_label(url)} ...")
        try:
            payload = fetch_json(session, url, sleep=sleep)
            monthly_games.append(extract_game_pdns(payload))
        except DownloadError as exc:
            failures.append(f"{url} ({exc})")
        sleep(delay)

    return DownloadResult(
        username=username,
        archives_total=len(archives),
        archives_processed=len(archives) - len(failures),
        games_downloaded=sum(map(len, monthly_games)),
        failed_requests=failures,
        pgn=join_games(monthly_games),
    )


def output_path_for(username: str, data_dir: Path | None = None) -> Path:
    """Default output path: data/raw/{username}.pgn."""
    return (data_dir or DEFAULT_DATA_DIR) / f"{username}.pgn"


def save_pgn(pgn: str, path: Path) -> None:
    """Write PGN string to file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pgn.rstrip() + "\n", encoding="utf-8")


def confirm_overwrite(
    path: Path,
    *,
    input_fn: Callable[[str], str] | None = None,
    force: bool = False,
) -> bool:
    """Prompt user or check force flag before overwriting an existing file."""
    if not path.exists() or force:
        return True
    ask = input_fn or input
    return ask(f"{path} already exists. Overwrite? [y/N] ").strip().lower() in {
        "y",
        "yes",
    }
