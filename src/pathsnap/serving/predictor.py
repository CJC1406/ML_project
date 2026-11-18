"""Predict one trip end to end: features -> candidate probabilities -> HMM/Viterbi -> connected route.

Prediction reads only .nodes, .arcs and .track. The optional evaluation step reads the reference
.route on the server and returns aggregate numbers only, never the reference arcs themselves.
"""
from __future__ import annotations

import os
import time

import numpy as np

from pathsnap.features.candidate_features import align_route, build_trip, load_trip, project
from pathsnap.matching.hmm import build_lattice, classifier_only, decode, reconstruct_route

DECODERS = ("hmm_viterbi", "classifier_only")


def route_scores(pred_route, reference, arc_len, arcs):
    P, G = set(map(int, pred_route)), set(map(int, reference))
    length = lambda s: float(arc_len[list(s)].sum()) if s else 0.0
    lp, lg, li = length(P), length(G), length(P & G)
    prec, rec = li / max(lp, 1e-9), li / max(lg, 1e-9)
    cont = (float(np.mean(arcs[pred_route[:-1], 1] == arcs[pred_route[1:], 0]))
            if len(pred_route) > 1 else 1.0)
    return dict(route_precision=prec, route_recall=rec,
                route_f1=2 * prec * rec / max(prec + rec, 1e-9),
                route_mismatch_fraction=(length(P - G) + length(G - P)) / max(lg, 1e-9),
                route_continuity=cont)


def predict_trip(trip_dir, model, hmm_config: dict, decoder: str = "hmm_viterbi"):
    """Return a dict with per-observation arcs, the reconstructed route and diagnostics."""
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}")
    timings = {}
    s = time.perf_counter()
    loaded = load_trip(str(trip_dir), with_route=False)
    _, nodes, arcs, track, _ = loaded
    feats = build_trip(str(trip_dir), with_labels=False, loaded=loaded)
    timings["features_ms"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    probs = model.predict_proba(feats["X"])[:, 1] if len(feats["X"]) else np.zeros(0)
    timings["model_ms"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    lat = build_lattice(feats, probs, nodes, arcs, track)
    if decoder == "hmm_viterbi":
        pred, breaks = decode(lat, hmm_config["alpha"], hmm_config["beta"])
    else:
        pred, breaks = classifier_only(lat), 0
    route, unjoined = reconstruct_route(lat.graph, pred, hmm_config.get("route_fill_limit_m", 2000.0))
    timings["decode_ms"] = (time.perf_counter() - s) * 1000
    no_cand = int(sum(len(c) == 0 for c in lat.states))
    return dict(nodes=nodes, arcs=lat.graph.arcs, track=track, arc_length=lat.graph.length,
                features=feats, probabilities=probs, predicted=pred, route=route,
                chain_breaks=int(breaks), unjoined_gaps=int(unjoined),
                observations_without_candidates=no_cand, timings=timings)


def evaluate_prediction(trip_dir, result) -> dict | None:
    """Server-side scores against the reference route, or None when the trip has no .route file."""
    tid = os.path.basename(os.path.normpath(str(trip_dir)))
    rp = os.path.join(str(trip_dir), f"{tid}.route")
    if not os.path.isfile(rp):
        return None
    reference = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    out = route_scores(result["route"], reference, result["arc_length"], result["arcs"])
    # observation-level Top-1 against the same monotone alignment used to derive training labels
    track, nodes, arcs = result["track"], result["nodes"], result["arcs"]
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    P = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    T = project(track[:, 0], track[:, 1], lon0, lat0)
    true_arc, true_d = align_route(T, P[arcs[:, 0]], P[arcs[:, 1]], reference)
    lab = true_d <= 50.0
    out["observation_top1"] = float(np.mean(result["predicted"][lab] == true_arc[lab])) if lab.any() else None
    out["labelled_observations"] = int(lab.sum())
    out["protocol"] = ("server-side comparison with the reference route; length-weighted route metrics; "
                       "Top-1 against the monotone route alignment (derived labels)")
    return out


def route_geojson(nodes, arcs, route):
    feats = []
    for order, a in enumerate(route):
        u, v = arcs[int(a)]
        feats.append({"type": "Feature", "properties": {"arc_id": int(a), "order": order},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[float(nodes[u, 0]), float(nodes[u, 1])],
                                                   [float(nodes[v, 0]), float(nodes[v, 1])]]}})
    return {"type": "FeatureCollection", "features": feats}


def gps_geojson(track):
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"kind": "gps", "points": int(len(track))},
        "geometry": {"type": "LineString", "coordinates": track[:, :2].astype(float).tolist()}}]}
