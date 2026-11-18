# PathSnap

PathSnap matches noisy GPS observations to a supplied road network and rebuilds the connected route the
vehicle drove. Each GPS reading gets a set of nearby candidate road arcs. A trained classifier scores how
likely each candidate is to be the true arc, and an HMM/Viterbi decoder then picks the most probable
connected sequence of arcs. The system runs on two laptops: a FastAPI backend (Laptop 1) and a React
evaluator UI (Laptop 2).

Course: UE24CS352A, PES University.

## What is in the repository

| Component | Where |
|---|---|
| Data loading, validation, preprocessing | `src/pathsnap/data/`, `scripts/preprocess_training.py`, `analysis/eda/`, `docs/eda/` |
| Candidate generation and feature engineering | `src/pathsnap/map_matching/`, `src/pathsnap/features/candidate_features.py`, `scripts/build_candidate_features.py` |
| Model training (logistic regression, KNN, decision tree, histogram gradient boosting, soft-vote and stacking ensembles) | `src/pathsnap/models/`, `scripts/knn/`, `scripts/ensemble/` |
| Model evaluation and comparison | `scripts/ensemble/compare_all_models.py`, `results/ensemble/model_comparison.csv` |
| Bayesian hyperparameter optimisation (Optuna, TPE) | `scripts/knn/knn_optuna.py`, `scripts/hgb/hgb_optuna.py`, `results/knn_tuned/`, `results/hgb_tuned/` |
| HMM and Viterbi route reconstruction | `src/pathsnap/matching/hmm.py`, `scripts/hmm/run_hmm.py`, `results/hmm/` |
| Exported model bundle (weights, HMM config, manifest with checksums) | `models/pathsnap_v1/`, `scripts/export_model.py` |
| Backend API | `src/pathsnap/api.py`, `src/pathsnap/serving/`, `scripts/run_backend.sh`, `docs/BACKEND_API.md` |
| Evaluator UI | `frontend/` |
| Folium maps and plots | `src/pathsnap/serving/maps.py`, `scripts/make_maps.py`, `scripts/*/plot_*.py` |
| Technical report | `docs/report/` |
| Tests | `tests/` |

Every split is by trip, never by individual GPS point. The evaluator partition is never read during
training, tuning or model selection. Reference `.route` files are read only by the backend to compute
aggregate metrics and are never sent to the UI.

## Installation

Requires Python 3.10+ and Node.js 18+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q
```

```bash
cd frontend && npm install
```

## Running the system on two laptops

Both laptops must be on the same network. Guest and university Wi-Fi often block traffic between
devices; a phone hotspot works as a fallback.

**Laptop 1, backend.** The trained bundle in `models/pathsnap_v1/` is loaded automatically.

```bash
PATHSNAP_EVALUATOR_ROOT="/path/to/evaluator_trips" scripts/run_backend.sh
```

The script binds `0.0.0.0:8000` and prints the address Laptop 2 should use. Each trip is one sub-folder
holding `.track`, `.nodes`, `.arcs` and `.route` files. Request logs appear in this terminal.

**Laptop 2, UI.**

```bash
cd frontend
echo "VITE_API_BASE_URL=http://<laptop-1-ip>:8000" > .env.local
npm run dev
```

Open http://localhost:5173, select a trip, choose a model and decoder, and run the prediction. The UI
shows the predicted route on a map and the evaluation metrics the backend computes. The Vite server
proxies `/api/*` to Laptop 1, so no CORS configuration is needed.

To check the backend from Laptop 2, run `curl http://<laptop-1-ip>:8000/health` or open
`http://<laptop-1-ip>:8000/docs`.

## Reproducing the experiments

Point the scripts at the dataset folder that contains `train/` and `student_test/`:

```bash
export PATHSNAP_DATA_ROOT="/path/to/Student folder"
```

Run in this order from the repository root:

```bash
python scripts/build_candidate_features.py        # candidate rows, labels, features -> data/processed/
python scripts/knn/run_knn.py                     # KNN baseline, CV and student test -> results/knn/
python scripts/knn/knn_optuna.py                  # KNN Bayesian tuning               -> results/knn_tuned/
python scripts/hgb/hgb_optuna.py                  # HGB Bayesian tuning               -> results/hgb_tuned/
python scripts/ensemble/compare_all_models.py     # all models + ensembles            -> results/ensemble/
python scripts/hmm/run_hmm.py                     # HMM tuning, classifier vs HMM     -> results/hmm/
python scripts/export_model.py                    # deployable bundle                 -> models/pathsnap_v1/
python scripts/make_maps.py                       # Folium maps                       -> results/maps/
```

All scripts use seed 42. Logs, configurations, trial histories and metrics are written next to their
results.

## Repository layout

```text
src/pathsnap/
  data/          loaders, schema, validation
  geo/           projections, distances, road geometry
  map_matching/  candidate generation
  features/      candidate features used by every model
  models/        model definitions (KNN, ensemble, estimators)
  matching/      HMM and Viterbi decoding
  training/      group-aware splits and evaluation helpers
  serving/       model bundle, single-trip predictor, Folium maps
  api.py         FastAPI backend
frontend/        React + Leaflet evaluator UI (Laptop 2)
scripts/         training, tuning, evaluation, export and plotting entry points
results/         metrics, logs, Optuna trials and plots from actual runs
models/          exported model bundle used by the backend
docs/            EDA findings, architecture, API, report
tests/           unit and API tests
```
