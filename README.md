# ChessEvolve

ChessEvolve is a CPU-friendly research project for ranking legal chess moves
with a small PyTorch model. It demonstrates:

- Stockfish-supervised move ranking.
- Verified recursive self-learning with an acceptance gate and rollback.
- Hard-label and soft-target knowledge distillation.
- A browser demo for human-vs-model and model-vs-model games.

The model ranks candidate moves supplied by `python-chess`; it is not a
replacement for Stockfish and is not presented as a competitive chess engine.

## Status

The research pipeline and browser demo are complete. The report, presentation,
backend, frontend, tests, and reproducibility instructions are included in this
repository. The large dataset, Stockfish binary, recommended checkpoint, and
other experiment outputs are supplied separately in the submission data
archive.

| Area | Status |
|---|---|
| Data preparation and Stockfish adapter | Complete |
| Baseline teacher training | Complete |
| Verified self-learning and rollback | Complete |
| Knowledge distillation | Complete |
| Evaluation and latency benchmarks | Complete |
| Human-vs-model demo | Complete |
| Model-vs-model demo | Complete |
| Documentation and release checklist | Complete |

## Results at a glance

The recommended mixed-loss B teacher has 136,577 trainable parameters. The
distilled students have 30,017 parameters, reducing parameter count by about
78%. The self-learning acceptance gate rejected a degraded candidate and
preserved the previous checkpoint. Historical checkpoints and comparisons are
retained in the experiment artifact bundle.

The current Phase 3.5 split contains 87,993 training, 19,101 development,
and 19,324 final-test records, partitioned by source game. Two invalid
final-test positions are excluded consistently during evaluation. Results are
move-ranking benchmarks against bounded Stockfish labels, not evidence of
general chess strength. See [`reports/final_report.md`](reports/final_report.md)
for the complete analysis.

## Repository layout

```text
.
├── backend/                  FastAPI inference service and model registry
├── frontend/                 React + Vite browser demo
├── src/chessevolve/          Reusable data, model, training, and evaluation code
├── scripts/                  Dataset, training, self-learning, and evaluation CLIs
├── tests/                    Core unit tests
├── backend/tests/            API tests
├── data/                     Input and processed datasets
├── experiments/results/      Saved checkpoints and evaluation artifacts
├── docs/                     Architecture and experiment documentation
└── reports/                  Final report and presentation
```

Generated files are intentionally excluded from Git. In particular,
`frontend/node_modules/`, `frontend/dist/`, Python caches, and model outputs
are ignored; `frontend/package-lock.json` is committed for reproducible npm
installs.

## Requirements and installation

- Python 3.10+ (the project has been tested with Python 3.14).
- Node.js 18+ and npm for the browser demo.
- A CPU is sufficient; no GPU is required.
- The research pipeline requires a Linux x86_64 Stockfish binary under
  `tools/stockfish/stockfish/`. The binary is excluded from GitHub because it
  exceeds GitHub's file limit; distribute it using the submission data archive
  and preserve `tools/stockfish/stockfish/Copying.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

cd frontend
npm ci
cd ..
```

Verify the engine:

```bash
python - <<'PY'
from pathlib import Path
from chessevolve.engine import EngineSettings, StockfishAdapter

path = Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal")
with StockfishAdapter(path) as engine:
    print(engine.version())
PY
```

## Run the research pipeline

Run the complete Python test suite:

```bash
source .venv/bin/activate
python -m pytest
```

The current repository collects 204 tests.

The main experiment commands are:

```bash
python scripts/train_baseline.py --help
python scripts/run_recursive_self_learning.py --help
python scripts/run_distillation.py --help
python scripts/evaluate_models.py --help
```

The checked-in experiment outputs are under `experiments/results/`. Large
checkpoints and generated outputs are ignored by default, so reproduce them
locally when needed.

## Run the browser demo

Start the backend from the repository root:

```bash
source .venv/bin/activate
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>.

The Vite development server proxies `/api` to FastAPI, so no frontend
environment variable is required. To use a different backend URL, set
`VITE_API_URL` before starting Vite.

The demo supports:

- **Human vs Model:** choose White or Black and play against any available
  checkpoint or the random opponent.
- **Model vs Model:** choose separate White and Black models, then autoplay,
  pause, or step through the game.
- Move history in SAN, game status, PGN export, temperature sampling, and
  top-three model analysis.

The backend API is:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/models` | List configured models and checkpoint availability |
| `POST` | `/move` | Rank legal moves and return the selected move |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/move \
  -H 'Content-Type: application/json' \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","model_id":"random","temperature":0}'
```

Run API tests:

```bash
python -m pytest backend/tests -q
```

## Documentation

- [`docs/concepts_guide.md`](docs/concepts_guide.md): key ideas and metrics.
- [`docs/architecture.md`](docs/architecture.md): end-to-end system design.
- [`docs/experimental_protocol.md`](docs/experimental_protocol.md): protocol,
  split policy, acceptance gate, and evaluation rules.
- [`docs/risk_and_failure_analysis.md`](docs/risk_and_failure_analysis.md):
  risks, limitations, and claims to avoid.
- [`docs/implementation_roadmap.md`](docs/implementation_roadmap.md): completed
  milestone record.
- [`reports/final_report.md`](reports/final_report.md): final results and
  interpretation.
- [`reports/presentation.md`](reports/presentation.md): presentation slides
  with speaker notes.

## Publishing checklist

Before pushing or publishing:

```bash
git status --short
git diff --check
source .venv/bin/activate
python -m pytest
cd frontend && npm ci && npm run build
```

Confirm that `git status` does not contain `frontend/node_modules/`, `frontend/dist/`,
Python caches, secrets, or locally generated checkpoints. Commit source,
configuration, lockfiles, documentation, and tests; publish the dataset,
Stockfish binary, and large experiment outputs separately if the hosting
platform has repository-size limits.

For the submission bundle, clone the repository and extract the archive at its
root:

```bash
tar -xzf chessevolve-submission-data.tar.gz
```

## License and external components

Check the repository's chosen license before publishing. Stockfish is an
external GPL-licensed component and must retain its license and attribution
when distributed. The frontend depends on React, Vite, `chess.js`, and
`react-chessboard`; review their licenses as part of the release process.
