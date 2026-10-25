# Data Directory

Do not commit downloaded PGNs, processed position corpora, or generated labels by default.

For every dataset, document:
- Source and licence/terms of use.
- Retrieval or generation date.
- Filtering and sampling procedure.
- Source-game grouping and split manifests.
- Dataset hash.
- Stockfish version and settings used to generate labels.

Split by source game before extracting positions to reduce leakage from related positions. Keep the final-test manifest separate from training and model-selection workflows.