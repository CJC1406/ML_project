# PathSnap Backend

Laptop 1 runs the backend; Laptop 2 runs the evaluator UI in `frontend/` (see `frontend/README.md`).
Both laptops must be on the same network.

## What it does

For a permitted trip ID the backend:

1. loads `.nodes`, `.arcs` and `.track` (never `.route`);
2. finds candidate arcs within 50 m of each GPS reading and computes the 15 candidate features;
3. scores every candidate with the selected model (default: the stacking ensemble);
4. decodes the most likely arc sequence with HMM/Viterbi, or ranks candidates independently if
   `decoder` is `classifier_only`;
5. joins consecutive arcs that do not touch with the shortest network path, giving a connected route;
6. if the trip folder on Laptop 1 contains a `.route` file, scores the prediction against it on the
   server and returns only the aggregate numbers.

## Start (Laptop 1)

Export the model bundle once (see `docs/report/TECHNICAL_REPORT.md`, "Reproduce"), then:

```bash
PATHSNAP_EVALUATOR_ROOT="/path/to/permitted/trips" scripts/run_backend.sh
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `PATHSNAP_EVALUATOR_ROOT` | `data/raw` | Folder whose sub-folders are the selectable trips |
| `PATHSNAP_MODEL_BUNDLE` | `models/pathsnap_v1` | Exported bundle (`scripts/export_model.py`) |
| `PATHSNAP_EVALUATE` | `1` | Set to `0` to skip server-side evaluation |
| `PATHSNAP_CORS_ORIGINS` | empty | Comma-separated origins, only if a browser calls the API directly |
| `PORT` / `PYTHON` | `8000` / `.venv/bin/python` | Server port and interpreter |

The script binds `0.0.0.0:8000` and prints the URL Laptop 2 should use. On macOS, allow incoming
connections for Python if the firewall prompts. If the bundle is missing or a file's checksum does
not match the manifest, the server still starts, `/health` reports `bundle_error`, and `/predict`
returns 503.

## Endpoints

- `GET /health` returns `{status, available_model_count, data_root_configured, trip_count, bundle, bundle_error, default_model, decoders}`.
- `GET /trips` returns `{trips: [...ids], partition}`. It lists only folders under the evaluator root that contain `.nodes`, `.arcs` and `.track`.
- `GET /models` returns `{models, default, decoders, labels}`. The models are every base model, `soft_vote` and `stacking`, with the default listed first.
- `GET /metrics` returns `{metrics: [...]}`, the student-test comparison stored in the bundle manifest.
- `POST /predict` takes `{"trip_id": "...", "model": "stacking", "decoder": "hmm_viterbi"}`. Only `trip_id` is required.
- `GET /map/{trip_id}?model=&decoder=` returns an interactive Folium HTML map with GPS readings, nearby roads and the predicted route. It never shows the reference route.

`/predict` response:

| Field | Meaning |
| --- | --- |
| `predicted_arc_ids` | Arc chosen for each GPS reading, in order (`-1` = no candidate) |
| `route_arc_ids` | Reconstructed, connected route |
| `route_geojson`, `gps_geojson` | Map layers for the route and the GPS trace |
| `sequence_method`, `hmm` | Decoder used and its parameters |
| `chain_breaks`, `unjoined_gaps`, `warnings` | Where the HMM restarted or the route could not be joined |
| `timings_ms`, `inference_time_ms` | Feature, model and decode times, and the total |
| `map_url` | Folium map for this prediction |
| `evaluation` | `null`, or server-side Top-1, route precision, recall, F1, mismatch fraction and continuity |

Invalid trip IDs are rejected with 422 and unknown trips or models with 404. Reference arc IDs and
reference geometry never appear in any response.

## Check from Laptop 2

```bash
curl http://<laptop-1-ip>:8000/health
```
