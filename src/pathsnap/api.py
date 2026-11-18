"""Laptop 1 FastAPI backend for prediction-only route reconstruction."""
from __future__ import annotations
import os, time
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.data.loader import load_trip
from pathsnap.geo.geometry import build_road_geometry
from pathsnap.map_matching.candidate_generation import generate_candidates
from pathsnap.ml.features import candidate_features, FEATURE_NAMES

TRAIN_ROOT = Path(os.getenv("PATHSNAP_EVALUATOR_ROOT", os.getenv("PATHSNAP_TRAIN_ROOT", "data/raw")))
MODEL_PATH = Path(os.getenv("PATHSNAP_LOGISTIC_MODEL", "experiments/local/exploratory_logistic/model/logistic_regression.joblib"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",") if x.strip()]
app = FastAPI(title="PathSnap API", version="0.2.0")
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictionRequest(BaseModel):
    model_name: str = "logistic_regression"
    sequence_method: str = "hmm_viterbi"


class ModelRegistry:
    def __init__(self):
        self.models = {}
        if MODEL_PATH.is_file():
            self.models["logistic_regression"] = {"model": joblib.load(MODEL_PATH), "artifact_path": str(MODEL_PATH), "available": True}

    def get(self, name):
        if name not in self.models: raise HTTPException(404, detail=f"Model unavailable: {name}")
        return self.models[name]


registry = ModelRegistry()


def permitted_trip_ids():
    return sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()) if TRAIN_ROOT.is_dir() else []


def geojson_for_arcs(road, arc_ids):
    features=[]
    for arc_id in arc_ids:
        matches=road.arcs.index[road.arcs.arc_id == arc_id].tolist()
        if not matches: continue
        geom=road.geographic_geometries[matches[0]]
        features.append({"type":"Feature","properties":{"arc_id":int(arc_id)},"geometry":{"type":"LineString","coordinates":[list(x) for x in geom.coords]}})
    return {"type":"FeatureCollection","features":features}


@app.get("/health")
def health():
    return {"status":"ok","available_model_count":len(registry.models),"data_root_configured":TRAIN_ROOT.is_dir()}


@app.get("/trips")
def trips():
    return {"partition":"configured_evaluator_root","trip_ids":permitted_trip_ids()}


@app.get("/models")
def models():
    return {"models":[{"model_name":name,"available":item["available"],"artifact_path":item["artifact_path"]} for name,item in registry.models.items()]}


@app.get("/models/metrics")
def model_metrics():
    metrics=MODEL_PATH.parent / "metrics.json"
    return {"metrics":metrics.read_text() if metrics.is_file() else None,"protocol":"EXPLORATORY provisional alignment metrics" if metrics.is_file() else None}


@app.post("/predict/{trip_id}")
def predict(trip_id: str, request: PredictionRequest):
    if trip_id not in permitted_trip_ids(): raise HTTPException(404, detail="Trip is not permitted")
    item=registry.get(request.model_name); started=time.perf_counter(); trip=load_trip(TRAIN_ROOT/trip_id); road=build_road_geometry(trip.nodes,trip.arcs); candidates=generate_candidates(trip.track,road,50,8)
    rows=candidate_features(trip.track,road,candidates); warnings=[]
    if not rows: warnings.append("no_candidates")
    frame=pd.DataFrame(rows); predicted=[]
    if not frame.empty:
        x=frame.reindex(columns=FEATURE_NAMES).replace([float("inf"),float("-inf")],float("nan")).fillna(0.0); model=item["model"]; probs=model.predict_proba(x); classes=list(model.classes_); positive=classes.index(1) if 1 in classes else probs.shape[1]-1; frame["probability"]=probs[:,positive]
        for _,group in frame.groupby("observation_index",sort=True): predicted.append(int(group.sort_values(["probability","distance_m","arc_id"],ascending=[False,True,True]).iloc[0].arc_id))
    if request.sequence_method == "hmm_viterbi": warnings.append("hmm_transition_model_unavailable_classifier_only_fallback")
    return {"trip_id":trip_id,"model_name":request.model_name,"sequence_method":"classifier_only_fallback" if request.sequence_method=="hmm_viterbi" else request.sequence_method,"predicted_arc_ids":predicted,"route_geojson":geojson_for_arcs(road,predicted),"inference_time_ms":(time.perf_counter()-started)*1000,"warnings":warnings,"evaluation":None}
