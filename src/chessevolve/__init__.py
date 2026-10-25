"""ChessEvolve package."""

__version__ = "0.1.0"

from .data import PositionRecord
from .splitting import (
    MANIFEST_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    DEFAULT_SPLIT_NAMES,
    DEFAULT_PROPORTIONS,
    SplitProportions,
    SplitResult,
    SplitError,
    split_by_source_game,
    build_manifest,
    write_manifest,
    write_split_manifest,
)

__all__ = [
    "PositionRecord",
    "MANIFEST_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_SPLIT_NAMES",
    "DEFAULT_PROPORTIONS",
    "SplitProportions",
    "SplitResult",
    "SplitError",
    "split_by_source_game",
    "build_manifest",
    "write_manifest",
    "write_split_manifest",
]
