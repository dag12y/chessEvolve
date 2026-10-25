"""Source-game-aware, reproducible dataset splitting for ChessEvolve.

This module never splits individual positions.  Every position that comes from
the same source game is assigned to exactly one of the configured splits,
which prevents leakage between related positions.

The split is deterministic: for a fixed seed and the same set of source
games, the assignment is always identical, regardless of the order in which
records are passed in.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .data import PositionRecord

MANIFEST_SCHEMA_VERSION = "1.0"
DATASET_SCHEMA_VERSION = "1.0"
DEFAULT_SPLIT_NAMES: Tuple[str, ...] = ("train", "dev", "test")
DEFAULT_PROPORTIONS: Tuple[float, ...] = (0.7, 0.15, 0.15)
SUPPORTED_SPLIT_NAMES = frozenset(DEFAULT_SPLIT_NAMES)
_PROPORTION_TOLERANCE = 1e-9


class SplitError(ValueError):
    """Raised when a split cannot be created safely."""


@dataclass(frozen=True)
class SplitProportions:
    """Validated split proportions for the standard train/dev/test splits."""

    train: float = 0.7
    dev: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        _validate_proportions(
            {"train": self.train, "dev": self.dev, "test": self.test}
        )

    def as_dict(self) -> Dict[str, float]:
        return {"train": self.train, "dev": self.dev, "test": self.test}

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.train, self.dev, self.test)


class SplitResult(dict):
    """A dict mapping split name -> list of records, plus split metadata.

    Behaves like an ordinary ``dict`` (``result["train"]`` works) while also
    exposing ``.manifest``, ``.assignments``, ``.seed`` and ``.proportions``.
    ``.proportions`` holds the *requested target* proportions; the manifest
    additionally reports the actual game and record proportions achieved.
    """

    def __init__(
        self,
        splits: Mapping[str, List[PositionRecord]],
        *,
        manifest: Dict[str, Any],
        assignments: Mapping[str, str],
        seed: int,
        proportions: Mapping[str, float],
    ) -> None:
        super().__init__({name: list(records) for name, records in splits.items()})
        self.manifest = dict(manifest)
        self.assignments = dict(assignments)
        self.seed = int(seed)
        self.proportions = dict(proportions)

    @property
    def records(self) -> List[PositionRecord]:
        """All records across every split, in split order."""
        result: List[PositionRecord] = []
        for records in self.values():
            result.extend(records)
        return result


def _clean_identifier(value: Any, field_name: str) -> str:
    """Strip and validate an identifier field; reject empty after normalization."""
    if not isinstance(value, str):
        raise SplitError(f"{field_name} must be a string, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise SplitError(f"{field_name} must not be empty or whitespace")
    return cleaned


def _validate_source_game_id(value: Any) -> str:
    return _clean_identifier(value, "source_game_id")


def _validate_position_id(value: Any) -> str:
    return _clean_identifier(value, "position_id")


def _validate_seed(seed: Any) -> int:
    """Accept integer seeds only; reject booleans, floats, strings and None."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError(f"seed must be an integer, got {type(seed).__name__}")
    return seed


def _validate_proportions(proportions: Mapping[str, float]) -> None:
    """Validate a split-name -> proportion mapping; raise SplitError on error.

    Rejects booleans and non-finite values (NaN, +/-inf).  Split names must be
    supported subset of train/dev/test.  Validated before any allocation.
    """
    if not proportions:
        raise SplitError("at least one split is required")
    for name in proportions:
        if not isinstance(name, str) or not name.strip():
            raise SplitError("split names must be non-empty strings")
        if name not in SUPPORTED_SPLIT_NAMES:
            raise SplitError(
                f"unsupported split name {name!r}; expected one of "
                f"{sorted(SUPPORTED_SPLIT_NAMES)}"
            )
    for name, value in proportions.items():
        if isinstance(value, bool):
            raise SplitError(f"proportion for {name!r} must be a number, got bool")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SplitError(
                f"proportion for {name!r} must be a number, got {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise SplitError(
                f"proportion for {name!r} must be finite, got {number}"
            )
        if number < 0:
            raise SplitError(
                f"proportion for {name!r} must not be negative, got {number}"
            )
    total = sum(float(value) for value in proportions.values())
    if not math.isfinite(total) or abs(total - 1.0) > _PROPORTION_TOLERANCE:
        raise SplitError(f"split proportions must sum to 1.0, got {total}")


def _normalize_record(record: PositionRecord) -> PositionRecord:
    """Return a copy of ``record`` with cleaned (stripped) identifiers.

    Every record returned by :func:`split_by_source_game` carries normalized
    identifiers so downstream joins/hashes treat " game_a " and "game_a" as
    the same game, consistent with the grouping and manifest logic.
    """
    cleaned_game_id = _validate_source_game_id(record.source_game_id)
    cleaned_position_id = _validate_position_id(record.position_id)
    if (
        cleaned_game_id == record.source_game_id
        and cleaned_position_id == record.position_id
    ):
        return record
    return dataclasses.replace(
        record,
        position_id=cleaned_position_id,
        source_game_id=cleaned_game_id,
    )


def _group_records_by_game(
    records: Iterable[PositionRecord],
) -> Dict[str, List[PositionRecord]]:
    """Group records by source game, validating ids (cleaned) as we go.

    Stored records carry normalized identifiers so the records returned to
    callers are consistent with grouping, assignments, and the manifest.
    """
    groups: Dict[str, List[PositionRecord]] = {}
    seen_position_ids: set = set()
    for record in records:
        if not isinstance(record, PositionRecord):
            raise SplitError(f"Expected PositionRecord, got {type(record).__name__}")
        normalized = _normalize_record(record)
        game_id = normalized.source_game_id
        position_id = normalized.position_id
        if position_id in seen_position_ids:
            raise SplitError(f"Duplicate position_id: {position_id!r}")
        seen_position_ids.add(position_id)
        groups.setdefault(game_id, []).append(normalized)
    return groups


def _allocate_games(
    game_ids: List[str],
    names: List[str],
    ratios: List[float],
    seed: int,
) -> Dict[str, str]:
    """Deterministically assign each source game to a split index.

    Uses largest-remainder allocation across *all* source games first, then
    repairs any empty positive-proportion split by moving a game from a split
    that has more than one (only possible when enough games exist).
    """
    if not game_ids:
        return {}

    n = len(game_ids)
    positive_indices = [i for i, r in enumerate(ratios) if r > 0]

    if n < len(positive_indices):
        raise SplitError(
            "Insufficient source games: need at least "
            f"{len(positive_indices)} game(s) for the non-empty splits, "
            f"got {n}"
        )

    rng = random.Random(seed)
    shuffled = list(game_ids)
    rng.shuffle(shuffled)

    # Largest-remainder allocation across all games.
    targets = {i: ratios[i] * n for i in positive_indices}
    floors = {i: int(targets[i]) for i in positive_indices}
    leftover = n - sum(floors.values())
    order = sorted(
        positive_indices,
        key=lambda i: targets[i] - floors[i],
        reverse=True,
    )
    for j in range(leftover):
        floors[order[j % len(order)]] += 1
    counts = dict(floors)

    # Repair empty positive-proportion splits by taking a game from a split
    # that has more than one.  When n >= len(positive_indices) such a donor
    # always exists whenever a split is empty.
    for idx in positive_indices:
        if counts[idx] != 0:
            continue
        donor = None
        for other in positive_indices:
            if other != idx and counts[other] > 1:
                donor = other
                break
        if donor is None:
            raise SplitError(
                "Insufficient source games to give every split at least one game"
            )
        counts[donor] -= 1
        counts[idx] = 1

    # Deal the shuffled games contiguously by the computed game counts.
    assignment: Dict[str, int] = {}
    pos = 0
    for idx in positive_indices:
        for _ in range(counts[idx]):
            if pos < len(shuffled):
                assignment[shuffled[pos]] = idx
                pos += 1

    # Guard against floating-point edge cases: any stragglers go to the
    # largest positive split.
    while pos < len(shuffled):
        largest = max(positive_indices, key=lambda i: ratios[i])
        assignment[shuffled[pos]] = largest
        pos += 1

    return {game: names[idx] for game, idx in assignment.items()}


def _validate_splits_for_manifest(splits: Mapping[str, Sequence[PositionRecord]]) -> None:
    """Validate manifest inputs: game isolation, split-field match, no dups."""
    seen_position_ids: set = set()
    seen_game_split: Dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            if not isinstance(record, PositionRecord):
                raise SplitError(
                    f"Expected PositionRecord, got {type(record).__name__}"
                )
            if record.split != split_name:
                raise SplitError(
                    f"record {record.position_id!r} has split {record.split!r} "
                    f"but is listed under {split_name!r}"
                )
            position_id = _validate_position_id(record.position_id)
            if position_id in seen_position_ids:
                raise SplitError(f"Duplicate position_id: {position_id!r}")
            seen_position_ids.add(position_id)
            game_id = _validate_source_game_id(record.source_game_id)
            if game_id in seen_game_split and seen_game_split[game_id] != split_name:
                raise SplitError(
                    f"source game {game_id!r} appears in multiple splits: "
                    f"{seen_game_split[game_id]!r} and {split_name!r}"
                )
            seen_game_split[game_id] = split_name


def _package_version() -> str:
    """Best-effort package version for the manifest (no hard metadata dependency)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("chessevolve")
        except PackageNotFoundError:
            pass
    except Exception:  # pragma: no cover - environments without importlib.metadata
        pass
    return "0.1.0"


def build_manifest(
    splits: Mapping[str, Sequence[PositionRecord]],
    *,
    seed: int,
    proportions: Mapping[str, float],
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION,
    dataset_schema_version: str = DATASET_SCHEMA_VERSION,
    created_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable manifest describing a split assignment.

    Validates its inputs directly so it is safe to call standalone.
    """
    seed = _validate_seed(seed)
    _validate_proportions(proportions)
    _validate_splits_for_manifest(splits)

    for name in splits:
        if name not in proportions:
            raise SplitError(f"missing proportion for split {name!r}")

    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()

    counts: Dict[str, Dict[str, int]] = {}
    game_ids: Dict[str, List[str]] = {}
    total_records = 0
    total_games = 0
    for name in splits:
        records = splits[name]
        games = sorted({_validate_source_game_id(r.source_game_id) for r in records})
        counts[name] = {
            "records": len(records),
            "source_games": len(games),
        }
        game_ids[name] = games
        total_records += len(records)
        total_games += len(games)

    actual_game_proportions = {
        name: (len(game_ids[name]) / total_games if total_games else 0.0)
        for name in splits
    }
    actual_record_proportions = {
        name: (len(splits[name]) / total_records if total_records else 0.0)
        for name in splits
    }

    return {
        "manifest_schema_version": manifest_schema_version,
        "dataset_schema_version": dataset_schema_version,
        "random_seed": int(seed),
        "split_proportions": {name: float(proportions[name]) for name in splits},
        "actual_game_proportions": actual_game_proportions,
        "actual_record_proportions": actual_record_proportions,
        "counts": counts,
        "source_game_ids": game_ids,
        "created_utc": created_utc,
        "implementation": {
            "package": "chessevolve",
            "version": _package_version(),
        },
    }


def _assign_split_to_record(record: PositionRecord, split_name: str) -> PositionRecord:
    """Return a copy of ``record`` carrying its assigned split label."""
    try:
        return dataclasses.replace(record, split=split_name)
    except ValueError as exc:
        raise SplitError(str(exc)) from exc


def split_by_source_game(
    records: Iterable[PositionRecord],
    *,
    seed: int = 0,
    proportions: Optional[Mapping[str, float]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> SplitResult:
    """Split ``records`` by source game and return a :class:`SplitResult`.

    Every source game is assigned to exactly one split, so all positions from
    the same game land in the same split.  The returned object behaves like a
    dict mapping split name -> list of records, and each record's ``split``
    field is set to its assigned split.

    Empty input is rejected by default (a split with no games is meaningless).

    Parameters
    ----------
    records:
        Iterable of :class:`PositionRecord` instances.
    seed:
        Integer random seed.  Fixed seed + identical input => identical split.
    proportions:
        Mapping of supported split name -> target proportion.  Keys must be a
        subset of ``train``/``dev``/``test``; values non-negative, finite and
        summing to 1.  Defaults to ``{"train": 0.7, "dev": 0.15, "test": 0.15}``.
    manifest_path:
        If given, the split manifest is written atomically to this path.

    Raises
    ------
    SplitError
        For invalid proportions/seeds, duplicate position ids, malformed
        source-game ids, empty input, unsupported split names, or too few
        source games for the requested splits.
    """
    seed = _validate_seed(seed)

    if proportions is None:
        proportions = dict(zip(DEFAULT_SPLIT_NAMES, DEFAULT_PROPORTIONS))

    # Canonical split order so the assignment does not depend on the order of
    # keys in the proportions mapping.
    names = [name for name in DEFAULT_SPLIT_NAMES if name in proportions]
    raw_values = [proportions[name] for name in names]
    _validate_proportions(dict(zip(names, raw_values)))
    values = [float(value) for value in raw_values]

    records_list = list(records)
    groups = _group_records_by_game(records_list)
    game_ids = sorted(groups.keys())

    if not game_ids:
        raise SplitError("no records to split")

    assignment = _allocate_games(game_ids, names, values, seed)

    splits: Dict[str, List[PositionRecord]] = {name: [] for name in names}
    assignments: Dict[str, str] = {}
    for game_id, split_name in assignment.items():
        assignments[game_id] = split_name
        for record in groups[game_id]:
            splits[split_name].append(_assign_split_to_record(record, split_name))

    proportion_map = {name: value for name, value in zip(names, values)}
    manifest = build_manifest(splits, seed=seed, proportions=proportion_map)

    result = SplitResult(
        splits,
        manifest=manifest,
        assignments=assignments,
        seed=seed,
        proportions=proportion_map,
    )

    if manifest_path is not None:
        write_manifest(manifest, manifest_path)

    return result


def write_manifest(manifest: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Atomically write a manifest dict as JSON and return the path written.

    Uses ``allow_nan=False`` so NaN/inf can never be serialized.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                manifest, handle, indent=2, sort_keys=True, default=str, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return destination


def write_split_manifest(
    path: Union[str, Path],
    splits: Mapping[str, Sequence[PositionRecord]],
    *,
    seed: int,
    proportions: Optional[Mapping[str, float]] = None,
) -> Path:
    """Build a manifest from splits and write it as JSON.

    ``proportions`` are the *requested target* proportions and must be supplied
    explicitly; they are never inferred from record counts (which would not be
    game proportions).  Actual game and record proportions are derived from the
    corresponding counts inside the manifest.
    """
    if proportions is None:
        raise SplitError("proportions must be provided to write_split_manifest")
    manifest = build_manifest(splits, seed=seed, proportions=proportions)
    return write_manifest(manifest, path)
