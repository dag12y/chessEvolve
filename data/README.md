# Data Directory

Do not commit downloaded PGNs, processed position corpora, or generated labels
by default. The final submission data archive preserves the paths needed by the
reproduction commands.

The final experiment uses `data/raw/dag12yi.pgn`, a Chess.com PGN export, and
the processed corpus under
`data/processed/dag12yi_phase35_labeled/`. Verify that the source PGN may be
redistributed under the applicable Chess.com terms before sharing it outside
the submission. The repository does not claim ownership of that source data.

For every dataset, document:
- Source and licence/terms of use.
- Retrieval or generation date.
- Filtering and sampling procedure.
- Source-game grouping and split manifests.
- Dataset hash.
- Stockfish version and settings used to generate labels.

Split by source game before extracting positions to reduce leakage from related positions. Keep the final-test manifest separate from training and model-selection workflows.