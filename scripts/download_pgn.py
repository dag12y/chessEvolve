#!/usr/bin/env python3
"""CLI script to download games for a Chess.com player via public API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for direct invocation
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import requests

from chessevolve.downloader import (
    DEFAULT_DATA_DIR,
    REQUEST_DELAY_SECONDS,
    USERNAME_PATTERN,
    DownloadError,
    confirm_overwrite,
    download_player_games,
    output_path_for,
    save_pgn,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download PGN archives for a Chess.com user.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Chess.com username to download games for. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where {username}.pgn will be saved.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Custom exact path to save the PGN file (overrides --output-dir).",
    )
    parser.add_argument(
        "--max-archives",
        type=int,
        default=None,
        help="Maximum number of monthly archives to download (most recent first).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help="Delay in seconds between sequential monthly archive requests.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        "--force",
        dest="force",
        action="store_true",
        help="Overwrite destination file without confirmation prompt.",
    )

    args = parser.parse_args()

    username = args.username
    if not username:
        try:
            username = input("Chess.com username: ").strip()
        except EOFError:
            print("error: no username provided", file=sys.stderr)
            return 2

    if not USERNAME_PATTERN.fullmatch(username):
        print(f"error: invalid Chess.com username {username!r}", file=sys.stderr)
        return 2

    path = args.output_path or output_path_for(username, args.output_dir)
    if not confirm_overwrite(path, force=args.force):
        print(f"Kept existing file: {path}")
        return 0

    try:
        with requests.Session() as session:
            result = download_player_games(
                username,
                session=session,
                delay=args.delay,
                max_archives=args.max_archives,
            )
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not result.pgn:
        print(f"error: no games found for {username}", file=sys.stderr)
        return 1

    save_pgn(result.pgn, path)
    print(
        f"Downloaded {result.games_downloaded} games from "
        f"{result.archives_processed}/{result.archives_total} archives."
    )
    print(f"Saved to {path}")

    if result.failed_requests:
        print("Some archives failed; output may be incomplete:", file=sys.stderr)
        for failure in result.failed_requests:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
