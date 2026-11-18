# PathSnap

PathSnap is a working, data-first map-matching system: noisy GPS observations are matched to a supplied road network and reconstructed as a connected route. The repository deliberately does not include fake trips, fake labels, or invented benchmark numbers.

## Current status

The real training partition has now been inspected and preprocessed. Completed work:

- strict loaders for `.track`, `.nodes`, `.arcs`, and `.route` inputs;
- schema validation with actionable errors;
- a dependency-free Viterbi decoder;
- a small FastAPI service surface (optional dependency);
- a browser frontend stub that calls the API;
- tests for validation, geometry, candidates, and Viterbi decoding; currently 10 tests pass locally.
- reusable EDA, road geometry, candidate generation, and Folium map utilities.

Not started yet: route/GPS alignment, supervised-label generation, ML training, five-model comparison, Optuna, and the production evaluator workflow.

Dataset-specific column assumptions remain isolated in `pathsnap/data/schema.py`. Put real trips under `data/raw/` only after inspecting their format. Ground-truth route files must stay on the backend/evaluation side and must never be sent to the frontend.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/python -m pathsnap.cli validate data/raw
.venv/bin/python -c 'from pathlib import Path; from src.eda import explore_dataset; explore_dataset(Path("data/raw"), Path("data/processed/eda.json"))'
.venv/bin/python scripts/preprocess_training.py
```

Set the dataset path locally before running data commands:

```bash
export PATHSNAP_TRAIN_ROOT="/absolute/path/to/Student folder/train"
```

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

Expected current result: `10 passed`.

The API and React application are intentionally separate: Laptop 1 runs the Python service; Laptop 2 runs the evaluator UI. See `docs/architecture.md`.

## Planned, evidence-backed phases

1. Inspect the supplied dataset and freeze its schema.
2. Add candidate generation and leakage-safe trip-level splits.
3. Train and compare five independently configured models.
4. Add Optuna studies with persisted configurations and results.
5. Integrate the trained scorer with HMM/Viterbi, Folium, FastAPI, and the evaluator UI.

No model metrics are reported until real data has been loaded and evaluated.

The dataset-aware preprocessing command reads `PATHSNAP_TRAIN_ROOT` (defaulting to
the local training path used during development) and writes ignored artifacts under
`data/processed/train/`: `track_features.csv`, projected node coordinates, arc tables,
per-trip CRS metadata, and `manifest.json`. It preserves elapsed timestamps, computes
time gaps, movement distance, valid GPS bearings, speeds, and large-gap flags. It does
not join `.route` files to observations or create supervised labels.

Candidate generation uses each trip's local projected CRS and a metric spatial index,
then ranks candidates by exact projected point-to-segment distance and optional
heading difference. It is deterministic and preserves observations with no candidates
by returning no rows for that observation. The selected EPSG/CRS is recorded per trip.

## Final repository shape

```text
src/pathsnap/
  data/          # raw-data boundary, schema, loaders, leakage checks
  features/      # spatial, temporal, direction and connectivity features
  models/        # five isolated model factories and training artifacts
  tuning/        # Optuna studies and immutable experiment manifests
  matching/      # candidate generation, HMM emissions, Viterbi decoding
  evaluation/    # backend-only labels, metrics and trip-level CV
  visualisation/ # Folium map generation
  api.py         # FastAPI prediction/evaluation boundary
frontend/        # React evaluator, no ground truth bundled
configs/         # versioned training and deployment configuration
data/raw/        # supplied inputs, gitignored
data/private/    # evaluator labels, backend-only, gitignored
experiments/     # configs, logs and results, no invented values
tests/           # unit and integration tests
docs/            # decisions and two-laptop deployment
```

## KNN model (candidate classifier)
```bash
export PATHSNAP_DATA_ROOT=/path/to/dataset      # folder containing train/ and student_test/
python3 scripts/build_candidate_features.py     # ~40 s, builds data/processed/candidate_features/
python3 scripts/knn/run_knn.py --quick          # fast demo (~2 min)
python3 scripts/knn/run_knn.py                  # full run: CV + student test + sequence metrics (~6 min)
python3 scripts/knn/knn_sweep.py                # k sweep for over/underfitting evidence (~10 min)
```
Outputs: `results/knn/` · Report section: `docs/report/knn.md`
