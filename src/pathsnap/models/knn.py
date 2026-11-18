"""Model: K-Nearest Neighbours candidate classifier (untuned baseline configuration).

Task for the model: for each (GPS observation, candidate arc) row predict P(candidate is the true arc).
Observation-level prediction = candidate with the highest probability (Top-1).

Protocol (matches the PathSnap Models & Training guideline):
  * 5-fold GroupKFold on the 80 training trips (groups = trip id -> no trip is split across folds).
  * Refit on all 80 training trips, evaluate once on the 10 student-testing trips.
  * Evaluator-testing trips are never touched.
Why scaling: KNN uses Euclidean distance in feature space, so features with large ranges (arc length in
metres, speed) would dominate unless standardised. StandardScaler is fitted inside the pipeline on the
training fold only (no leakage).
Why subsampling: KNN stores the whole training set and every prediction scans it, so we keep every
SUBSAMPLE-th GPS observation (all of its candidate rows) for fitting. Recorded in the results.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix)
import joblib

from pathsnap.features.candidate_features import FEATURES, load_cache

SEED = 42
SUBSAMPLE = 10
CONFIG = dict(n_neighbors=25, weights="distance", metric="minkowski", p=2, algorithm="auto")
# Optional per-feature multipliers applied AFTER standardisation (feature-weighted distance).
# None = plain KNN (all features count equally). Set by scripts/knn/knn_optuna.py for the tuned model.
FEATURE_WEIGHTS = None
MODEL_NAME = "KNN"
THRESHOLD = 0.5     # row-level decision threshold for P(true arc); Top-1 does not depend on it


def scale_cols(X, w):
    """Multiply standardised columns by weights w (module-level so the pipeline can be pickled)."""
    return X * w


def stack(trips, subsample=1):
    Xs, ys, gs = [], [], []
    for g, tr in enumerate(trips):
        keep = tr["row_labelled"] & (tr["obs"] % subsample == 0)
        Xs.append(tr["X"][keep]); ys.append(tr["y"][keep]); gs.append(np.full(keep.sum(), g))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def connected(arcs_from, arcs_to, a, b):
    return a == b or arcs_to[a] == arcs_from[b]


def evaluate(model, trips, arc_tables=None, obs_every=1, probs=None):
    """Row-level classification metrics + observation-level Top-1 + candidate miss rate + continuity."""
    y_all, p_all = [], []
    top1_hit = top1_n = miss = lab_obs = 0
    cont_ok = cont_n = 0
    t0 = time.perf_counter()
    for tr in trips:
        if obs_every > 1:   # score a regular subset of observations (all their candidates) to save time
            sel = tr["obs"] % obs_every == 0
            tr = dict(tr, X=tr["X"][sel], y=tr["y"][sel], obs=tr["obs"][sel], cand=tr["cand"][sel],
                      row_labelled=tr["row_labelled"][sel],
                      true_arc=np.where(np.arange(len(tr["true_arc"])) % obs_every == 0, tr["true_arc"], -1))
            arc_tables_use = None
        else:
            arc_tables_use = arc_tables
        if probs is not None and tr["trip"] in probs:
            p = probs[tr["trip"]]
        else:
            p = model.predict_proba(tr["X"])[:, 1]
            if probs is not None:
                probs[tr["trip"]] = p
        m = tr["row_labelled"]
        y_all.append(tr["y"][m]); p_all.append(p[m])
        obs, cand, true_arc = tr["obs"], tr["cand"], tr["true_arc"]
        # best candidate per observation
        order = np.lexsort((-p, obs))
        first = np.r_[True, obs[order][1:] != obs[order][:-1]]
        best_obs, best_arc = obs[order][first], cand[order][first]
        pred = np.full(int(tr["n_obs"]), -1); pred[best_obs] = best_arc
        lab = true_arc >= 0
        lab_obs += lab.sum()
        has_true = np.zeros(int(tr["n_obs"]), bool); has_true[obs[tr["y"] == 1]] = True
        miss += (lab & ~has_true).sum()
        top1_hit += (pred[lab] == true_arc[lab]).sum(); top1_n += lab.sum()
        if arc_tables_use is not None:
            af, at = arc_tables_use[tr["trip"]]
            seq = pred[pred >= 0]
            seq = seq[np.r_[True, seq[1:] != seq[:-1]]]          # collapse repeats
            if len(seq) > 1:
                ok = (at[seq[:-1]] == af[seq[1:]])
                cont_ok += ok.sum(); cont_n += len(ok)
    infer_s = time.perf_counter() - t0
    y = np.concatenate(y_all); p = np.concatenate(p_all); yhat = (p >= THRESHOLD).astype(int)
    return dict(
        accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0), macro_f1=f1_score(y, yhat, average="macro"),
        log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
        top1_accuracy=top1_hit / max(top1_n, 1), candidate_miss_rate=miss / max(lab_obs, 1),
        route_continuity=(cont_ok / cont_n) if cont_n else None,
        inference_time_s=infer_s, n_rows=len(y),
        confusion_matrix=confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    )


def arc_table(root, split, tid):
    arcs = np.loadtxt(os.path.join(root, split, tid, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
    return arcs[:, 0], arcs[:, 1]


def make_model():
    steps = [StandardScaler()]
    if FEATURE_WEIGHTS is not None:
        steps.append(FunctionTransformer(scale_cols, kw_args=dict(w=np.asarray(FEATURE_WEIGHTS, float))))
    steps.append(KNeighborsClassifier(n_jobs=-1, **CONFIG))
    return make_pipeline(*steps)


def main(root, cache_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(SEED)
    train = load_cache(cache_dir, "train")
    test = load_cache(cache_dir, "student_test")
    tables = {str(tr["trip"]): arc_table(root, "train", str(tr["trip"])) for tr in train}
    tables.update({str(tr["trip"]): arc_table(root, "student_test", str(tr["trip"])) for tr in test})
    for tr in train + test:
        tr["trip"] = str(tr["trip"])

    # ---- 5-fold group CV on training trips
    folds = []
    gkf = GroupKFold(n_splits=5)
    trip_idx = np.arange(len(train))
    for k, (tr_i, va_i) in enumerate(gkf.split(trip_idx, groups=trip_idx)):
        X, y, _ = stack([train[i] for i in tr_i], SUBSAMPLE)
        m = make_model(); s = time.perf_counter(); m.fit(X, y); fit_s = time.perf_counter() - s
        r = evaluate(m, [train[i] for i in va_i], tables, obs_every=3)
        r.update(fold=k, fit_time_s=fit_s, n_train_rows=len(y),
                 val_trips=[train[i]["trip"] for i in va_i])
        folds.append(r)
        print(f"fold {k}: macroF1={r['macro_f1']:.4f} top1={r['top1_accuracy']:.4f} "
              f"logloss={r['log_loss']:.4f} ({r['inference_time_s']:.0f}s)", flush=True)

    # ---- refit on all 80 training trips, evaluate on student testing
    X, y, _ = stack(train, SUBSAMPLE)
    model = make_model(); s = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - s
    probs = {}
    test_res = evaluate(model, test, tables, probs=probs)
    test_res.update(fit_time_s=fit_s, n_train_rows=len(y))
    per_trip = []
    for tr in test:
        r1 = evaluate(model, [tr], tables, probs=probs)
        per_trip.append(dict(trip=tr["trip"], top1_accuracy=r1["top1_accuracy"],
                             macro_f1=r1["macro_f1"], route_continuity=r1["route_continuity"]))

    np.savez_compressed(os.path.join(out_dir, "knn_student_test_probs.npz"), **probs)  # for HMM stage
    keys = ["accuracy", "precision", "recall", "macro_f1", "log_loss", "top1_accuracy",
            "candidate_miss_rate", "route_continuity", "inference_time_s", "fit_time_s"]
    cv_summary = {k: dict(mean=float(np.mean([f[k] for f in folds])),
                          std=float(np.std([f[k] for f in folds])))
                  for k in keys if folds[0][k] is not None}
    results = dict(model=MODEL_NAME, family="Instance-based", config=CONFIG, threshold=THRESHOLD,
                   feature_weights=(None if FEATURE_WEIGHTS is None else
                                    dict(zip(FEATURES, map(float, FEATURE_WEIGHTS)))), subsample_every=SUBSAMPLE,
                   scaler="StandardScaler", features=FEATURES, seed=SEED,
                   cv_protocol="5-fold GroupKFold by trip on 80 training trips; validation scored on every 3rd observation (all candidates)",
                   cv_folds=folds, cv_summary=cv_summary, student_test=test_res,
                   student_test_per_trip=per_trip)
    with open(os.path.join(out_dir, "knn_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    joblib.dump(model, os.path.join(out_dir, "knn_model.joblib"))
    print(json.dumps({"cv": cv_summary, "student_test": {k: test_res.get(k) for k in keys}}, indent=2, default=float))
    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
