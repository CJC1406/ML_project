from pathlib import Path
import json
import pandas as pd
from pathsnap.data.loader import discover_trip_directories, load_trip
from pathsnap.data.validator import validate_trip, parse_timestamps
from pathsnap.geo.road_network import build_graph, graph_statistics


def explore_dataset(root: Path, output: Path) -> dict:
    """Write data-derived inventory JSON; produces no output when no trips exist."""
    trips = [load_trip(p) for p in discover_trip_directories(root)]
    report = {"trip_count": len(trips), "trips": []}
    for trip in trips:
        validation = validate_trip(trip)
        graph = build_graph(trip.nodes, trip.arcs)
        times = parse_timestamps(trip.track.timestamp)
        gaps = times.diff().dropna()
        report["trips"].append({"trip_id": trip.trip_id, "track_rows": len(trip.track), "node_rows": len(trip.nodes), "arc_rows": len(trip.arcs), "route_rows": len(trip.route), "file_valid": validation.valid, "errors": validation.errors, "warnings": validation.warnings, "timestamp_gap_seconds": {"min": gaps.min() if len(gaps) else None, "max": gaps.max() if len(gaps) else None}, "graph": graph_statistics(graph)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
