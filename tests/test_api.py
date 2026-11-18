"""End-to-end API checks on a synthetic trip and a small model bundle (no dataset needed)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pathsnap.api import create_app
from pathsnap.features.candidate_features import FEATURES, build_trip
from pathsnap.serving.bundle import BundleError, load_bundle, save_bundle

# East-west street of 6 nodes (~111 m apart) with arcs in both directions, plus a side street.
NODES = np.array([[0.001 * i, 0.0] for i in range(6)] + [[0.002, 0.001]])
ARCS = np.array([[i, i + 1] for i in range(5)] + [[i + 1, i] for i in range(5)] + [[2, 6], [6, 2]])
ROUTE = np.array([0, 1, 2, 3, 4])                      # eastbound along the street


def write_trip(root, tid="00000001", with_route=True):
    d = root / tid
    d.mkdir(parents=True)
    t = np.arange(0, 56)
    track = np.c_[0.0001 + t * 0.00009, np.full(len(t), 0.00004), t]   # ~10 m/s, 4 m north of the street
    np.savetxt(d / f"{tid}.nodes", NODES)
    np.savetxt(d / f"{tid}.arcs", ARCS, fmt="%d")
    np.savetxt(d / f"{tid}.track", track)
    if with_route:
        np.savetxt(d / f"{tid}.route", ROUTE, fmt="%d")
    return d


@pytest.fixture()
def setup(tmp_path):
    trips = tmp_path / "trips"
    trip_dir = write_trip(trips)
    write_trip(trips, "00000002", with_route=False)
    f = build_trip(str(trip_dir))
    X, y = f["X"], f["y"]
    base = {"decision_tree": DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y),
            "logistic_regression": LogisticRegression(max_iter=500).fit(X, y)}
    meta = LogisticRegression().fit(np.c_[np.linspace(-3, 3, 20), np.linspace(-3, 3, 20)],
                                    np.r_[np.zeros(10), np.ones(10)])
    bundle = tmp_path / "bundle"
    save_bundle(bundle, base, meta, dict(alpha=1.0, beta=10.0, max_gap_s=60.0),
                dict(default_model="stacking", metrics=[{"model": "stacking", "accuracy": 1.0}]))
    return TestClient(create_app(bundle, trips, evaluate=True)), bundle


def test_health_trips_models(setup):
    client, _ = setup
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["available_model_count"] == 4 and h["trip_count"] == 2
    assert client.get("/trips").json()["trips"] == ["00000001", "00000002"]
    models = client.get("/models").json()
    assert models["models"][0] == "stacking" and "hmm_viterbi" in models["decoders"]
    assert client.get("/metrics").json()["metrics"][0]["model"] == "stacking"


def test_predict_returns_connected_route_without_labels(setup):
    client, _ = setup
    r = client.post("/predict", json={"trip_id": "00000001"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sequence_method"] == "hmm_viterbi" and body["model"] == "stacking"
    assert len(body["predicted_arc_ids"]) == body["observations"] == 56
    route = body["route_arc_ids"]
    assert all(ARCS[a][1] == ARCS[b][0] for a, b in zip(route, route[1:]))   # connected
    assert body["evaluation"]["route_f1"] > 0.9
    payload = json.dumps(body)
    assert "reference" not in body["evaluation"] and '"route":' not in payload


def test_predict_without_reference_and_classifier_only(setup):
    client, _ = setup
    body = client.post("/predict", json={"trip_id": "00000002", "model": "decision_tree",
                                         "decoder": "classifier_only"}).json()
    assert body["evaluation"] is None and body["hmm"] is None


def test_rejects_unknown_trip_model_decoder_and_bad_ids(setup):
    client, _ = setup
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "00000001", "model": "nope"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "00000001", "decoder": "x"}).status_code == 422
    assert client.post("/predict", json={"trip_id": "../etc"}).status_code == 422


def test_map_endpoint_serves_folium_html(setup):
    client, _ = setup
    html = client.get("/map/00000001").text
    assert "leaflet" in html.lower() and "HMM/Viterbi route" in html


def test_missing_or_tampered_bundle_fails_clearly(setup, tmp_path):
    client, bundle = setup
    (bundle / "hmm_config.json").write_text("{}")
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(bundle)
    empty = TestClient(create_app(tmp_path / "missing", tmp_path / "trips"))
    assert empty.get("/health").json()["available_model_count"] == 0
    assert empty.post("/predict", json={"trip_id": "00000001"}).status_code == 503
