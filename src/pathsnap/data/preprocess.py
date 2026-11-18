"""Dataset-aware, label-preserving preprocessing for the training partition."""
from pathlib import Path
import json
import math
import pandas as pd
from .loader import discover_trip_directories, load_trip
from .validator import parse_timestamps, validate_trip
from ..geo.distance import bearing_deg, haversine_m
from ..geo.projection import choose_trip_projection, project_xy


def preprocess_training(root: Path, output: Path) -> dict:
    """Create per-trip track/network features and a machine-readable manifest.

    Reference routes are intentionally not joined to track rows. They remain in the
    source partition for later alignment research and evaluation only.
    """
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_root": str(root), "trip_count": 0, "trips": []}
    for directory in discover_trip_directories(root):
        trip = load_trip(directory)
        report = validate_trip(trip)
        if not report.valid:
            manifest["trips"].append({"trip_id": trip.trip_id, "valid": False, "errors": report.errors})
            continue
        trip_out = output / trip.trip_id
        if (trip_out / "metadata.json").is_file():
            manifest["trips"].append(json.loads((trip_out / "metadata.json").read_text(encoding="utf-8")))
            manifest["trip_count"] += 1
            continue
        trip_out.mkdir(parents=True, exist_ok=True)
        track = trip.track.copy()
        track["elapsed_seconds"] = parse_timestamps(track["timestamp"])
        track["delta_seconds"] = track["elapsed_seconds"].diff()
        track["movement_m"] = 0.0
        track["gps_bearing_deg"] = float("nan")
        for i in range(1, len(track)):
            prev, row = track.iloc[i - 1], track.iloc[i]
            track.loc[track.index[i], "movement_m"] = haversine_m(prev.longitude, prev.latitude, row.longitude, row.latitude)
            dt = track.loc[track.index[i], "delta_seconds"]
            if pd.notna(dt) and 0 < dt <= 60 and track.loc[track.index[i], "movement_m"] >= 1.0:
                track.loc[track.index[i], "gps_bearing_deg"] = bearing_deg(prev.longitude, prev.latitude, row.longitude, row.latitude)
        track["speed_mps"] = track["movement_m"].div(track["delta_seconds"].where(track["delta_seconds"] > 0))
        track["large_gap"] = track["delta_seconds"].gt(60).fillna(False)
        track.to_csv(trip_out / "track_features.csv", index=False)
        projection = choose_trip_projection(trip.nodes.longitude.tolist(), trip.nodes.latitude.tolist())
        nodes = trip.nodes.copy()
        projected = [project_xy(projection, lon, lat) for lon, lat in zip(nodes.longitude, nodes.latitude)]
        nodes["x_m"], nodes["y_m"] = zip(*projected)
        nodes.to_csv(trip_out / "nodes_projected.csv", index=False)
        arcs = trip.arcs.copy()
        arcs.to_csv(trip_out / "arcs.csv", index=False)
        metadata = {"trip_id": trip.trip_id, "source_crs": projection.source_crs, "target_crs": projection.target_crs, "epsg": projection.epsg, "track_rows": len(track), "node_rows": len(nodes), "arc_rows": len(arcs), "route_rows": len(trip.route)}
        (trip_out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest["trips"].append(metadata | {"valid": True})
        manifest["trip_count"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
