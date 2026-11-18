"""Laptop 1 FastAPI backend: trip ID in -> candidate scoring -> HMM/Viterbi -> reconstructed route out.

Environment
  PATHSNAP_EVALUATOR_ROOT   folder whose sub-folders are the permitted trips (default data/raw)
  PATHSNAP_MODEL_BUNDLE     exported bundle from scripts/export_model.py (default models/pathsnap_v1)
  PATHSNAP_EVALUATE         "1" (default) to return server-side metrics when a trip has a .route file
  PATHSNAP_CORS_ORIGINS     optional comma-separated origins (the Vite proxy makes CORS unnecessary)

Reference .route files never leave the server: predictions read only .nodes/.arcs/.track, and the
evaluation block holds aggregate numbers only. See docs/BACKEND_API.md.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pathsnap.serving.bundle import BundleError, load_bundle
from pathsnap.serving.predictor import (DECODERS, evaluate_prediction, gps_geojson, predict_trip,
                                        route_geojson)

REQUIRED = (".nodes", ".arcs", ".track")


class PredictionRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    model: str | None = None
    decoder: str = "hmm_viterbi"


def create_app(bundle_path=None, trips_root=None, evaluate: bool | None = None) -> FastAPI:
    bundle_path = Path(bundle_path or os.getenv("PATHSNAP_MODEL_BUNDLE", "models/pathsnap_v1"))
    trips_root = Path(trips_root or os.getenv("PATHSNAP_EVALUATOR_ROOT",
                                              os.getenv("PATHSNAP_TRAIN_ROOT", "data/raw")))
    if evaluate is None:
        evaluate = os.getenv("PATHSNAP_EVALUATE", "1") != "0"
    try:
        models, hmm_config, manifest = load_bundle(bundle_path)
        bundle_error = None
    except BundleError as e:
        models, hmm_config, manifest, bundle_error = {}, None, {}, str(e)
    default_model = manifest.get("default_model", "stacking") if models else None
    model_names = ([default_model] + [m for m in models if m != default_model]) if models else []
    maps: dict = {}

    app = FastAPI(title="PathSnap API", version="1.0.0")
    origins = [x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",") if x.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["*"])

    def permitted_trips():
        if not trips_root.is_dir():
            return []
        return sorted(p.name for p in trips_root.iterdir()
                      if p.is_dir() and all((p / f"{p.name}{ext}").is_file() for ext in REQUIRED))

    def run(trip_id, model_name, decoder):
        if not models:
            raise HTTPException(503, detail=f"No model bundle loaded: {bundle_error}")
        if trip_id not in permitted_trips():
            raise HTTPException(404, detail=f"Trip is not permitted: {trip_id}")
        model_name = model_name or default_model
        if model_name not in models:
            raise HTTPException(404, detail=f"Model unavailable: {model_name}")
        if decoder not in DECODERS:
            raise HTTPException(422, detail=f"decoder must be one of {list(DECODERS)}")
        return model_name, predict_trip(trips_root / trip_id, models[model_name], hmm_config, decoder)

    @app.get("/health")
    def health():
        return {"status": "ok", "available_model_count": len(models),
                "data_root_configured": trips_root.is_dir(), "trip_count": len(permitted_trips()),
                "bundle": str(bundle_path), "bundle_error": bundle_error,
                "default_model": default_model, "decoders": list(DECODERS)}

    @app.get("/trips")
    def trips():
        return {"trips": permitted_trips(), "partition": "configured_evaluator_root"}

    @app.get("/models")
    def list_models():
        return {"models": model_names, "default": default_model, "decoders": list(DECODERS),
                "labels": manifest.get("model_labels", {})}

    @app.get("/metrics")
    def metrics():
        return {"metrics": manifest.get("metrics", [])}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        started = time.perf_counter()
        model_name, r = run(request.trip_id, request.model, request.decoder)
        warnings = []
        if r["observations_without_candidates"]:
            warnings.append(f"{r['observations_without_candidates']} GPS readings had no road within "
                            f"{manifest.get('candidates', {}).get('radius_m', 50)} m")
        if r["chain_breaks"]:
            warnings.append(f"HMM chain restarted {r['chain_breaks']} times (time gap or unreachable roads)")
        if r["unjoined_gaps"]:
            warnings.append(f"{r['unjoined_gaps']} route gaps could not be joined by a network path")
        evaluation = evaluate_prediction(trips_root / request.trip_id, r) if evaluate else None
        maps[(request.trip_id, model_name, request.decoder)] = r
        return {
            "trip_id": request.trip_id, "model": model_name, "sequence_method": request.decoder,
            "predicted_arc_ids": [int(a) for a in r["predicted"]],
            "route_arc_ids": [int(a) for a in r["route"]],
            "route_geojson": route_geojson(r["nodes"], r["arcs"], r["route"]),
            "gps_geojson": gps_geojson(r["track"]),
            "observations": int(len(r["track"])),
            "chain_breaks": r["chain_breaks"], "unjoined_gaps": r["unjoined_gaps"],
            "hmm": ({k: hmm_config[k] for k in ("alpha", "beta", "max_gap_s")}
                    if request.decoder == "hmm_viterbi" else None),
            "timings_ms": r["timings"],
            "inference_time_ms": (time.perf_counter() - started) * 1000,
            "map_url": f"/map/{request.trip_id}?model={model_name}&decoder={request.decoder}",
            "warnings": warnings, "evaluation": evaluation,
        }

    @app.get("/map/{trip_id}", response_class=HTMLResponse)
    def folium_map(trip_id: str, model: str | None = None, decoder: str = "hmm_viterbi"):
        from pathsnap.serving.maps import nearby_arcs, trip_map
        model_name = model or default_model
        r = maps.get((trip_id, model_name, decoder))
        if r is None:
            model_name, r = run(trip_id, model_name, decoder)
        label = "HMM/Viterbi route" if decoder == "hmm_viterbi" else "Classifier-only route"
        fmap = trip_map(r["nodes"], r["arcs"], r["track"],
                        {f"{label} ({model_name})": (r["route"], "hmm" if decoder == "hmm_viterbi" else "classifier")},
                        network_arcs=nearby_arcs(r["features"]), title=f"Trip {trip_id}: {label}")
        return HTMLResponse(fmap.get_root().render())

    return app


app = create_app()
