"""Tests for source-game-aware splitting and manifest writing."""

from __future__ import annotations

import json
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import pytest

from chessevolve.data import PositionRecord
from chessevolve.splitting import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_PROPORTIONS,
    DEFAULT_SPLIT_NAMES,
    MANIFEST_SCHEMA_VERSION,
    SplitError,
    SplitProportions,
    SplitResult,
    build_manifest,
    split_by_source_game,
    write_manifest,
    write_split_manifest,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def make_record(position_id: str, source_game_id: str, split: str = "train") -> PositionRecord:
    return PositionRecord(
        position_id=position_id,
        source_game_id=source_game_id,
        fen=STARTING_FEN,
        candidate_moves=["e7e5", "c7c5", "d7d5"],
        split=split,
    )


def make_games(game_count: int, positions_per_game: int = 2) -> list[PositionRecord]:
    records: list[PositionRecord] = []
    for g in range(game_count):
        for p in range(positions_per_game):
            records.append(make_record(f"g{g}p{p}", f"game_{g:03d}"))
    return records


# --- Identifier normalization ----------------------------------------------

def test_source_game_id_whitespace_stripped_and_normalized() -> None:
    """Whitespace variants of the same game must land in the same split.

    Records returned to callers must carry cleaned identifiers so downstream
    joins/hashes treat " game_a " and "game_a" as the same game.
    """
    records = [
        make_record("p1", " game_a "),
        make_record("p2", "game_a"),
        make_record("p3", "  game_a\t"),
    ]
    result = split_by_source_game(records, seed=0, proportions={"train": 1.0})
    # Returned records carry cleaned identifiers, not the raw whitespace variants.
    assert all(r.source_game_id == "game_a" for r in result.records)
    assert all(r.position_id == f"p{i}" for i, r in enumerate(result.records, start=1))
    assert len(result["train"]) == 3


def test_whitespace_variant_game_cannot_span_splits() -> None:
    """A game with whitespace-variant ids must not appear in multiple splits."""
    records = [
        make_record("p1", " game_a "),
        make_record("p2", "game_a"),
        make_record("p3", "game_b"),
        make_record("p4", " game_b "),
        make_record("p5", "game_c"),
    ]
    result = split_by_source_game(records, seed=0)
    games_per_split = {
        name: {r.source_game_id for r in recs}
        for name, recs in result.items()
    }
    # Each game lives in exactly one split.
    for name in games_per_split:
        for other in games_per_split:
            if name != other:
                assert not (games_per_split[name] & games_per_split[other])
    assert "game_a" in set().union(*games_per_split.values())
    assert "game_b" in set().union(*games_per_split.values())
    # All records carry cleaned identifiers.
    assert all(r.source_game_id in {"game_a", "game_b", "game_c"} for r in result.records)
    assert sum(len(v) for v in result.values()) == 5


def test_whitespace_variant_duplicate_position_id_rejected() -> None:
    records = [
        make_record(" p1 ", "g1"),
        make_record("p1", "g2"),
    ]
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=0)


def test_whitespace_variant_duplicate_position_id_rejected() -> None:
    records = [
        make_record(" p1 ", "g1"),
        make_record("p1", "g2"),
    ]
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=0)


def test_empty_source_game_id_rejected() -> None:
    for bad_id in ("", "   ", "\t\n"):
        records = [make_record("p1", bad_id)]
        with pytest.raises(SplitError):
            split_by_source_game(records, seed=0)


def test_empty_position_id_rejected() -> None:
    records = [make_record("   ", "g1")]
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=0)


def test_non_string_source_game_id_rejected() -> None:
    records = [make_record("p1", 42)]  # type: ignore[arg-type]
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=0)


def test_non_string_position_id_rejected() -> None:
    records = [make_record(7, "g1")]  # type: ignore[arg-type]
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=0)


# --- Game-level isolation / completeness -----------------------------------

def test_game_ids_do_not_overlap_across_splits() -> None:
    records = make_games(20)
    result = split_by_source_game(records, seed=7)

    seen: dict[str, str] = {}
    for split_name, split_records in result.items():
        for record in split_records:
            gid = record.source_game_id.strip()
            assert gid not in seen or seen[gid] == split_name
            seen[gid] = split_name

    assert len(seen) == 20
    games_per_split = {
        name: {r.source_game_id.strip() for r in recs}
        for name, recs in result.items()
    }
    for name in games_per_split:
        for other in games_per_split:
            if name != other:
                assert not (games_per_split[name] & games_per_split[other])


def test_every_record_assigned_exactly_once() -> None:
    records = make_games(15, positions_per_game=3)
    result = split_by_source_game(records, seed=3)

    all_records = result.records
    assert len(all_records) == len(records) == 45
    input_ids = {r.position_id.strip() for r in records}
    output_ids = {r.position_id.strip() for r in all_records}
    assert input_ids == output_ids
    assert len(output_ids) == sum(len(v) for v in result.values())


def test_record_split_field_matches_assignment() -> None:
    records = make_games(10)
    result = split_by_source_game(records, seed=1)

    for split_name, split_records in result.items():
        for record in split_records:
            assert record.split == split_name


# --- Determinism -----------------------------------------------------------

def test_same_seed_produces_identical_split() -> None:
    records = make_games(30)
    first = split_by_source_game(records, seed=42)
    second = split_by_source_game(records, seed=42)

    assert first.assignments == second.assignments
    assert {k: [r.position_id for r in v] for k, v in first.items()} == {
        k: [r.position_id for r in v] for k, v in second.items()
    }
    for key in (
        "manifest_schema_version",
        "random_seed",
        "split_proportions",
        "actual_game_proportions",
        "actual_record_proportions",
        "counts",
        "source_game_ids",
        "implementation",
    ):
        assert first.manifest[key] == second.manifest[key]


def test_input_order_does_not_change_split() -> None:
    records = make_games(30)
    import random as _random

    shuffled = list(records)
    _random.Random(99).shuffle(shuffled)

    first = split_by_source_game(records, seed=5)
    second = split_by_source_game(shuffled, seed=5)
    assert first.assignments == second.assignments


def test_different_seeds_produce_different_splits() -> None:
    records = make_games(40)
    first = split_by_source_game(records, seed=1).assignments
    second = split_by_source_game(records, seed=2).assignments
    assert first != second


# --- Configurable proportions ----------------------------------------------

def test_default_proportions_are_valid() -> None:
    proportions = SplitProportions().as_dict()
    assert proportions == {"train": 0.7, "dev": 0.15, "test": 0.15}
    assert abs(sum(proportions.values()) - 1.0) < 1e-9


def test_custom_proportions() -> None:
    records = make_games(20)
    result = split_by_source_game(
        records, seed=4, proportions={"train": 0.6, "dev": 0.2, "test": 0.2}
    )
    assert set(result) == {"train", "dev", "test"}
    assert result.manifest["split_proportions"] == {
        "train": 0.6,
        "dev": 0.2,
        "test": 0.2,
    }


def test_subset_proportions() -> None:
    records = make_games(10)
    result = split_by_source_game(
        records, seed=1, proportions={"train": 0.5, "test": 0.5}
    )
    assert set(result) == {"train", "test"}
    assert len(result.records) == 20


def test_equivalent_proportions_different_key_order() -> None:
    records = make_games(20)
    a = split_by_source_game(
        records, seed=3, proportions={"train": 0.7, "dev": 0.15, "test": 0.15}
    )
    b = split_by_source_game(
        records, seed=3, proportions={"test": 0.15, "train": 0.7, "dev": 0.15}
    )
    # Same seed + same target proportions => same assignment (order-independent).
    assert a.assignments == b.assignments


# --- Proportion validation -------------------------------------------------

def test_proportion_rejects_boolean() -> None:
    records = make_games(5)
    with pytest.raises(SplitError):
        split_by_source_game(
            records, seed=1, proportions={"train": True, "dev": 0.0, "test": 0.0}
        )


def test_proportion_rejects_nan() -> None:
    records = make_games(5)
    with pytest.raises(SplitError):
        split_by_source_game(
            records, seed=1, proportions={"train": float("nan"), "dev": 0.0, "test": 0.0}
        )


def test_proportion_rejects_infinity() -> None:
    records = make_games(5)
    for bad in (float("inf"), float("-inf")):
        with pytest.raises(SplitError):
            split_by_source_game(
                records, seed=1, proportions={"train": bad, "dev": 0.0, "test": 0.0}
            )


def test_proportion_validation_errors_before_allocation() -> None:
    """Invalid proportions raise even when there are plenty of games."""
    records = make_games(100)
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=1, proportions={"train": 0.5, "dev": 0.5, "test": 0.5})


def test_split_proportions_dataclass_validation() -> None:
    assert SplitProportions() == SplitProportions(train=0.7, dev=0.15, test=0.15)
    with pytest.raises(ValueError):
        SplitProportions(train=0.5, dev=0.5, test=0.5)  # sums to 1.5
    with pytest.raises(ValueError):
        SplitProportions(train=-0.1, dev=0.6, test=0.5)  # negative
    with pytest.raises(ValueError):
        SplitProportions(train=float("nan"), dev=0.0, test=0.0)


# --- Seed validation -------------------------------------------------------

def test_seed_rejects_boolean() -> None:
    records = make_games(3)
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=True)  # type: ignore[arg-type]


def test_seed_rejects_float() -> None:
    records = make_games(3)
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=1.5)  # type: ignore[arg-type]


def test_seed_rejects_string() -> None:
    records = make_games(3)
    with pytest.raises(SplitError):
        split_by_source_game(records, seed="42")  # type: ignore[arg-type]


def test_seed_rejects_none() -> None:
    records = make_games(3)
    with pytest.raises(SplitError):
        split_by_source_game(records, seed=None)  # type: ignore[arg-type]


def test_valid_seed_reproduced_in_result_and_manifest() -> None:
    records = make_games(10)
    result = split_by_source_game(records, seed=123)
    assert result.seed == 123
    assert result.manifest["random_seed"] == 123


# --- Allocation amounts ----------------------------------------------------

def test_default_70_15_15_with_20_games() -> None:
    records = make_games(20, positions_per_game=1)
    result = split_by_source_game(records, seed=0)
    game_counts = {name: len(result.manifest["source_game_ids"][name]) for name in result}
    assert game_counts == {"train": 14, "dev": 3, "test": 3}


def test_default_70_15_15_with_3_games() -> None:
    records = make_games(3, positions_per_game=1)
    result = split_by_source_game(records, seed=0)
    game_counts = {name: len(result.manifest["source_game_ids"][name]) for name in result}
    assert game_counts == {"train": 1, "dev": 1, "test": 1}


def test_insufficient_source_games_raises() -> None:
    records = make_games(2)  # only 2 games for 3 non-empty splits
    with pytest.raises(SplitError) as exc_info:
        split_by_source_game(records, seed=1)
    assert "Insufficient source games" in str(exc_info.value)


def test_zero_proportion_split_is_empty() -> None:
    records = make_games(8)
    result = split_by_source_game(
        records, seed=2, proportions={"train": 1.0, "dev": 0.0, "test": 0.0}
    )
    assert result["train"] and not result["dev"] and not result["test"]
    assert result.manifest["counts"]["dev"]["records"] == 0


def test_single_game_lands_in_train_only() -> None:
    records = [make_record("p1", "only_game")]
    result = split_by_source_game(
        records, seed=0, proportions={"train": 1.0, "dev": 0.0, "test": 0.0}
    )
    assert result["train"] == [dataclasses_replace(records[0], split="train")]
    assert not result["dev"] and not result["test"]


# --- Empty input / invalid names ------------------------------------------

def test_empty_input_rejected() -> None:
    with pytest.raises(SplitError):
        split_by_source_game([], seed=0)


def test_unsupported_split_name_rejected() -> None:
    records = make_games(3)
    with pytest.raises(SplitError):
        split_by_source_game(
            records, seed=0, proportions={"development": 0.5, "final": 0.5}
        )


# --- Manifest correctness --------------------------------------------------

def test_manifest_has_required_fields() -> None:
    records = make_games(12)
    result = split_by_source_game(records, seed=11)
    manifest = result.manifest

    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert manifest["random_seed"] == 11
    assert manifest["split_proportions"] == {
        "train": 0.7,
        "dev": 0.15,
        "test": 0.15,
    }
    assert set(manifest["counts"]) == {"train", "dev", "test"}
    for split_name in ("train", "dev", "test"):
        assert set(manifest["counts"][split_name]) == {"records", "source_games"}
        assert manifest["counts"][split_name]["records"] == len(result[split_name])
        assert manifest["counts"][split_name]["source_games"] == len(
            manifest["source_game_ids"][split_name]
        )
    all_manifest_games = {
        g for games in manifest["source_game_ids"].values() for g in games
    }
    assert all_manifest_games == {f"game_{g:03d}" for g in range(12)}
    assert manifest["created_utc"].endswith("+00:00") or "Z" in manifest["created_utc"]
    assert manifest["implementation"]["package"] == "chessevolve"


def test_manifest_actual_game_and_record_proportions() -> None:
    """Actual proportions derived from counts; may differ from targets."""
    # Game 0 has 10 positions, games 1-19 have 1 each => 29 positions over 20 games.
    records: list[PositionRecord] = []
    for p in range(10):
        records.append(make_record(f"g0p{p}", "game_000"))
    for g in range(1, 20):
        records.append(make_record(f"g{g}p0", f"game_{g:03d}"))
    result = split_by_source_game(records, seed=0)
    manifest = result.manifest

    game_props = manifest["actual_game_proportions"]
    record_props = manifest["actual_record_proportions"]
    total_games = sum(c["source_games"] for c in manifest["counts"].values())
    total_records = sum(c["records"] for c in manifest["counts"].values())
    for name in result:
        assert abs(game_props[name] - manifest["counts"][name]["source_games"] / total_games) < 1e-12
        assert abs(record_props[name] - manifest["counts"][name]["records"] / total_records) < 1e-12
    # With unequal positions per game, record proportions differ from game proportions.
    assert record_props != game_props


def test_manifest_source_game_ids_match_assignments() -> None:
    records = make_games(16)
    result = split_by_source_game(records, seed=5)
    for split_name, game_ids in result.manifest["source_game_ids"].items():
        assert set(game_ids) == {
            game for game, name in result.assignments.items() if name == split_name
        }


def test_manifest_json_serialization_round_trip(tmp_path: Path) -> None:
    records = make_games(12)
    result = split_by_source_game(records, seed=6)

    path = tmp_path / "nested" / "dir" / "manifest.json"
    written = write_manifest(result.manifest, path)
    assert written == path
    assert path.exists()

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded["manifest_schema_version"] == result.manifest["manifest_schema_version"]
    assert loaded["random_seed"] == result.manifest["random_seed"]
    assert loaded["split_proportions"] == result.manifest["split_proportions"]
    assert loaded["actual_game_proportions"] == result.manifest["actual_game_proportions"]
    assert loaded["actual_record_proportions"] == result.manifest["actual_record_proportions"]
    assert loaded["counts"] == result.manifest["counts"]
    assert loaded["source_game_ids"] == result.manifest["source_game_ids"]
    assert loaded["implementation"] == result.manifest["implementation"]


def test_write_manifest_strict_json_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    with pytest.raises((ValueError, OverflowError)):
        write_manifest({"x": float("nan")}, path)


def test_write_manifest_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "manifest.json"
    write_manifest({"key": "value"}, path)
    assert path.exists()


def test_write_split_manifest_requires_proportions(tmp_path: Path) -> None:
    records = make_games(6)
    result = split_by_source_game(records, seed=0)
    path = tmp_path / "split_manifest.json"
    with pytest.raises(SplitError):
        write_split_manifest(path, result, seed=0)  # type: ignore[call-arg]


def test_write_split_manifest_round_trip(tmp_path: Path) -> None:
    records = make_games(9)
    result = split_by_source_game(records, seed=8)
    path = tmp_path / "split_manifest.json"
    write_split_manifest(
        path,
        result,
        seed=8,
        proportions=result.proportions,
    )
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["source_game_ids"] == result.manifest["source_game_ids"]
    assert loaded["counts"] == result.manifest["counts"]


def test_build_manifest_counts_match_splits() -> None:
    records = make_games(7, positions_per_game=2)
    result = split_by_source_game(records, seed=0)
    manifest = build_manifest(result, seed=0, proportions=result.proportions)
    for split_name in result:
        assert manifest["counts"][split_name]["records"] == len(result[split_name])
        assert manifest["counts"][split_name]["source_games"] == len(
            {r.source_game_id for r in result[split_name]}
        )


def test_build_manifest_rejects_game_in_multiple_splits() -> None:
    rec_train = make_record("p1", "g1", split="train")
    rec_dev = make_record("p2", "g1", split="dev")  # same game, different position
    with pytest.raises(SplitError) as exc_info:
        build_manifest(
            {"train": [rec_train], "dev": [rec_dev]},
            seed=0,
            proportions={"train": 0.5, "dev": 0.5},
        )
    assert "multiple splits" in str(exc_info.value).lower()


def test_build_manifest_rejects_split_field_mismatch() -> None:
    rec = make_record("p1", "g1", split="train")
    with pytest.raises(SplitError):
        build_manifest(
            {"dev": [rec]},
            seed=0,
            proportions={"dev": 1.0},
        )


def test_build_manifest_rejects_duplicate_position_id() -> None:
    a = make_record("p1", "g1", split="train")
    b = make_record("p1", "g2", split="dev")
    with pytest.raises(SplitError):
        build_manifest(
            {"train": [a], "dev": [b]},
            seed=0,
            proportions={"train": 0.5, "dev": 0.5},
        )


def test_build_manifest_rejects_malformed_record() -> None:
    with pytest.raises(SplitError):
        build_manifest(
            {"train": ["not a record"]},  # type: ignore[list-item]
            seed=0,
            proportions={"train": 1.0},
        )


def test_split_result_dict_and_attribute_access() -> None:
    records = make_games(6)
    result = split_by_source_game(records, seed=0)
    assert isinstance(result, dict)
    assert "train" in result
    assert result["train"] is result.get("train")
    assert result.seed == 0
    assert result.proportions == {"train": 0.7, "dev": 0.15, "test": 0.15}
    assert len(result.records) == 12


def test_no_absolute_paths_or_secrets_in_manifest() -> None:
    records = make_games(6)
    manifest = split_by_source_game(records, seed=0).manifest
    serialized = json.dumps(manifest)
    assert "/home" not in serialized
    assert "secret" not in serialized.lower()
    assert "password" not in serialized.lower()
