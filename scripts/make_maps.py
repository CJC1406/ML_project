"""Folium maps for the report: every student-test trip with GPS, nearby roads, reference route,
classifier-only route and HMM/Viterbi route (default model of the exported bundle).

Usage (repo root):
    export PATHSNAP_DATA_ROOT="/path/to/Student folder"
    python scripts/make_maps.py [--screenshots trip_id ...]
Outputs: results/maps/<trip>.html (interactive) and, with --screenshots, results/maps/<trip>.png
(headless Google Chrome; map tiles need internet access).
"""
import json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pathsnap.serving.bundle import load_bundle
from pathsnap.serving.maps import nearby_arcs, trip_map
from pathsnap.serving.predictor import evaluate_prediction, predict_trip

OUT = REPO / "results" / "maps"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def data_root():
    if os.environ.get("PATHSNAP_DATA_ROOT"):
        return Path(os.environ["PATHSNAP_DATA_ROOT"])
    sys.exit("Set PATHSNAP_DATA_ROOT to the folder that contains student_test/")


def main(shots):
    OUT.mkdir(parents=True, exist_ok=True)
    models, hmm_cfg, manifest = load_bundle(REPO / "models" / "pathsnap_v1")
    name = manifest["default_model"]
    split = data_root() / "student_test"
    summary = {}
    for trip_dir in sorted(p for p in split.iterdir() if p.is_dir()):
        tid = trip_dir.name
        hmm = predict_trip(trip_dir, models[name], hmm_cfg, "hmm_viterbi")
        clf = predict_trip(trip_dir, models[name], hmm_cfg, "classifier_only")
        reference = np.loadtxt(trip_dir / f"{tid}.route", dtype=np.int64, ndmin=1)
        e_h, e_c = evaluate_prediction(trip_dir, hmm), evaluate_prediction(trip_dir, clf)
        title = (f"Student-test trip {tid}: route F1 {e_c['route_f1']:.3f} classifier only, "
                 f"{e_h['route_f1']:.3f} with HMM/Viterbi")
        fmap = trip_map(hmm["nodes"], hmm["arcs"], hmm["track"], {
            "Reference route (ground truth)": (reference, "reference"),
            "Classifier-only route": (clf["route"], "classifier"),
            "HMM/Viterbi route": (hmm["route"], "hmm"),
        }, network_arcs=nearby_arcs(hmm["features"]), title=title)
        fmap.save(str(OUT / f"{tid}.html"))
        summary[tid] = dict(classifier_only=e_c, hmm_viterbi=e_h)
        print(title, flush=True)
    (OUT / "maps_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for tid in shots:
        png = OUT / f"{tid}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--window-size=1400,900", "--virtual-time-budget=15000",
                        f"--screenshot={png}", (OUT / f"{tid}.html").as_uri()],
                       check=True, capture_output=True, timeout=120)
        print("screenshot", png)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[args.index("--screenshots") + 1:] if "--screenshots" in args else [])
