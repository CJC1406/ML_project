# PathSnap Backend

Laptop 1 serves prediction requests with:

```bash
PATHSNAP_EVALUATOR_ROOT="/path/to/permitted/trips" \
PATHSNAP_LOGISTIC_MODEL="experiments/local/exploratory_logistic/model/logistic_regression.joblib" \
PATHSNAP_CORS_ORIGINS="http://laptop-2-host:5173" \
.venv/bin/uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` reports data-root and loaded-model status.
- `GET /trips` returns only IDs in the configured permitted partition.
- `GET /models` lists genuinely loaded artifacts.
- `GET /models/metrics` returns available metric artifacts and their provisional protocol label.
- `POST /predict/{trip_id}` accepts `{ "model_name": "logistic_regression", "sequence_method": "hmm_viterbi" }`.

The current backend performs classifier-only ranking and returns an explicit warning when HMM/Viterbi transition probabilities are unavailable. Reference `.route` files are never read by the prediction endpoint and are not included in GeoJSON.

Laptop 2 must call `http://<laptop-1-lan-ip>:8000`; the LAN address is environment-specific and must not be hard-coded.
