# PathSnap

PathSnap is a working, data-first map-matching system: noisy GPS observations are matched to a supplied road network and reconstructed as a connected route. The repository deliberately does not include fake trips, fake labels, or invented benchmark numbers.

## Current milestone

This first slice is runnable before the competition dataset arrives:

- strict loaders for `.track`, `.nodes`, `.arcs`, and `.route` inputs;
- schema validation with actionable errors;
- a dependency-free Viterbi decoder;
- a small FastAPI service surface (optional dependency);
- a browser frontend stub that calls the API;
- tests for validation and route reconstruction.
- reusable EDA, road geometry, candidate generation, and Folium map utilities.

Dataset-specific column assumptions remain isolated in `pathsnap/data/schema.py`. Put real trips under `data/raw/` only after inspecting their format. Ground-truth route files must stay on the backend/evaluation side and must never be sent to the frontend.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
.venv/bin/python -m pathsnap.cli validate data/raw
.venv/bin/python -c 'from pathlib import Path; from src.eda import explore_dataset; explore_dataset(Path("data/raw"), Path("data/processed/eda.json"))'
```

The API and React application are intentionally separate: Laptop 1 runs the Python service; Laptop 2 runs the evaluator UI. See `docs/architecture.md`.

## Planned, evidence-backed phases

1. Inspect the supplied dataset and freeze its schema.
2. Add candidate generation and leakage-safe trip-level splits.
3. Train and compare five independently configured models.
4. Add Optuna studies with persisted configurations and results.
5. Integrate the trained scorer with HMM/Viterbi, Folium, FastAPI, and the evaluator UI.

No model metrics are reported until real data has been loaded and evaluated.

Phase 2 candidate generation uses a spatial index for coarse nearby-segment queries,
then ranks candidates by metre-scale point-to-segment distance and optional heading
difference. It is deterministic and preserves observations with no candidates by
returning no rows for that observation. The current distance conversion is a local
equirectangular approximation; a region-specific projected CRS must be selected after
the real dataset is inspected.

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
