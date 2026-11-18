"""PathSnap - KNN model, end to end, from the terminal.

Usage (from the repository root):
    export PATHSNAP_DATA_ROOT="/path/to/Student folder"     # folder containing train/ and student_test/
    python3 scripts/knn/run_knn.py            # full run: features -> 5-fold CV -> student test (~6-8 min)
    python3 scripts/knn/run_knn.py --quick    # print the saved results in results/knn/ (instant)

Outputs: results/knn/ (metrics JSON, plots, model, logs). Feature cache: data/processed/candidate_features/.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)


def _data_root():
    if os.environ.get("PATHSNAP_DATA_ROOT"):
        return os.environ["PATHSNAP_DATA_ROOT"]
    if os.environ.get("PATHSNAP_TRAIN_ROOT"):            # team convention: points at .../train
        return os.path.dirname(os.path.normpath(os.environ["PATHSNAP_TRAIN_ROOT"]))
    sys.exit("Set PATHSNAP_DATA_ROOT to the folder that contains train/ and student_test/")


DATA = _data_root()
CACHE = os.path.join(REPO, "data", "processed", "candidate_features")
RESULTS = os.path.join(REPO, "results", "knn")


def line(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


def show_data():
    from pathsnap.features.candidate_features import load_trip
    line("STEP 1 - The raw data")
    for split in ("train", "student_test"):
        print(f"{split:13s}: {len(os.listdir(os.path.join(DATA, split)))} trips")
    tid, nodes, arcs, track, route = load_trip(os.path.join(DATA, "train", "00000010"))
    print(f"\nExample trip {tid}:")
    print(f"  .nodes : {len(nodes):6d} road points      first row (lon, lat): {nodes[0]}")
    print(f"  .arcs  : {len(arcs):6d} road pieces       first row (from_node, to_node): {arcs[0]}")
    print(f"  .track : {len(track):6d} GPS readings     first row (lon, lat, t): {track[0]}")
    print(f"  .route : {len(route):6d} true arc ids      first 5: {route[:5]}")


def show_features():
    from pathsnap.features.candidate_features import cache_all, load_cache, FEATURES
    line("STEP 2 - Candidates + labels + features for every trip")
    s = time.time()
    cache_all(DATA, CACHE)
    print(f"(feature build done in {time.time()-s:.0f}s; cached in {CACHE})")
    tr = load_cache(CACHE, "train")[0]
    print(f"\nTrip {tr['trip']}: {int(tr['n_obs'])} GPS points -> {len(tr['obs'])} (point, candidate road) rows")
    print(f"Average candidates per GPS point: {tr['n_cand'].mean():.1f}")
    print(f"Rows labelled 1 (true road): {int(tr['y'].sum())}   labelled 0: {int((tr['y']==0).sum())}")
    print("\nFirst GPS point's candidate rows (label 1 = the true road):")
    hdr = ["arc"] + FEATURES[:6] + ["label"]
    print("  " + " ".join(f"{h:>14s}" for h in hdr))
    for r in np.where(tr["obs"] == 0)[0]:
        vals = [tr["cand"][r]] + list(tr["X"][r, :6]) + [tr["y"][r]]
        print("  " + " ".join(f"{v:14.2f}" if isinstance(v, (float, np.floating)) else f"{v:14d}" for v in vals))


def show_results():
    r = json.load(open(os.path.join(RESULTS, "knn_results.json")))
    line("STEP 3 - 5-fold cross-validation (grouped by trip, 80 training trips)")
    print(f"Model config: {r['config']}, StandardScaler, train on every {r['subsample_every']}th GPS point")
    print(f"{'fold':>5} {'macroF1':>9} {'top1':>8} {'logloss':>9} {'train rows':>11}")
    for f in r["cv_folds"]:
        print(f"{f['fold']:>5} {f['macro_f1']:9.4f} {f['top1_accuracy']:8.4f} {f['log_loss']:9.4f} {f['n_train_rows']:11d}")
    c = r["cv_summary"]
    print(f"{'mean':>5} {c['macro_f1']['mean']:9.4f} {c['top1_accuracy']['mean']:8.4f} {c['log_loss']['mean']:9.4f}")
    print(f"{'std':>5} {c['macro_f1']['std']:9.4f} {c['top1_accuracy']['std']:8.4f} {c['log_loss']['std']:9.4f}")

    line("STEP 4 - Student-testing trips (10 trips, never used for training)")
    t = r["student_test"]
    for k in ["accuracy", "precision", "recall", "macro_f1", "log_loss", "top1_accuracy",
              "candidate_miss_rate", "route_continuity", "fit_time_s", "inference_time_s"]:
        print(f"  {k:22s} {t[k]:.4f}")
    cm = t["confusion_matrix"]
    print(f"\n  Confusion matrix (rows = actual, cols = predicted; 0 = not true road, 1 = true road)")
    print(f"           pred 0    pred 1\n  actual 0 {cm[0][0]:8d} {cm[0][1]:8d}\n  actual 1 {cm[1][0]:8d} {cm[1][1]:8d}")
    print(f"\n  {'trip':>10} {'top1':>8} {'macroF1':>9} {'continuity':>11}")
    for p in r["student_test_per_trip"]:
        print(f"  {p['trip']:>10} {p['top1_accuracy']:8.4f} {p['macro_f1']:9.4f} {p['route_continuity']:11.4f}")
    print(f"\nPlots: {RESULTS}/knn_cv_folds.png, knn_confusion_matrix.png, knn_student_test_per_trip.png")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    show_data()
    if not quick:
        show_features()
        line("Training KNN (5 folds + final fit + student test) ... about 6-8 min")
        import pathsnap.models.knn as knn
        knn.main(DATA, CACHE, RESULTS)
        import knn_sequence_metrics
        line("Sequence-level performance (KNN only, no HMM) on student-testing trips")
        knn_sequence_metrics.main(DATA, CACHE, RESULTS)
        os.system(f'"{sys.executable}" "{os.path.join(HERE, "plot_knn.py")}" "{RESULTS}"')
    elif os.path.isdir(CACHE):
        show_features()
    show_results()
    seq = os.path.join(RESULTS, "knn_sequence_metrics.json")
    if os.path.exists(seq):
        m = json.load(open(seq))["mean"]
        line("STEP 5 - Sequence-level performance (KNN only, no HMM), mean over student-test trips")
        for k in ("route_precision", "route_recall", "route_f1", "route_mismatch_fraction", "route_continuity"):
            print(f"  {k:24s} {m[k]:.4f}")
    sw = os.path.join(RESULTS, "knn_sweep.csv")
    if os.path.exists(sw):
        import csv
        line("STEP 6 - Sensitivity to k (train vs validation macro F1) -> over/under-fitting evidence")
        print(f"  {'k':>4} {'weights':>9} {'train F1':>9} {'val F1':>8} {'val top1':>9} {'val logloss':>12}")
        for r in csv.DictReader(open(sw)):
            print(f"  {r['k']:>4} {r['weights']:>9} {float(r['train_macro_f1_mean']):9.4f} "
                  f"{float(r['val_macro_f1_mean']):8.4f} {float(r['val_top1_accuracy_mean']):9.4f} "
                  f"{float(r['val_log_loss_mean']):12.4f}")
