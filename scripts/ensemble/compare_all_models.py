"""Train every PathSnap candidate model under one protocol and build ensembles from them.

Same rows, same features, same folds for every model, so the numbers are directly comparable:
  * features   pathsnap.features.candidate_features (15 features, 50 m radius, <=10 candidates)
  * CV         5-fold GroupKFold by trip on the 80 training trips (identical folds to the KNN study);
               validation scored on every 3rd GPS observation (all of its candidates)
  * test       refit on all 80 training trips, score once on the 10 student-test trips
  * evaluator  trips are never read
Ensembles:
  * soft_vote  mean of the four base probabilities (no fitting)
  * stacking   logistic regression on logit(base probabilities). Trained on out-of-fold (OOF) base
               probabilities only, so it never sees a probability a base model gave its own training
               rows. Its CV score uses the same 5 trip folds over the OOF table.
The exported "ensemble" is whichever ensemble has the higher mean CV Top-1. Student-test numbers
play no part in that choice.

Usage (repo root):
    export PATHSNAP_DATA_ROOT="/path/to/Student folder"      # contains train/ and student_test/
    python scripts/ensemble/compare_all_models.py
Outputs: results/ensemble/
"""
import csv, json, os, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "knn"))

import joblib
import pathsnap.models.knn as K
from pathsnap.features.candidate_features import FEATURES, cache_all, load_cache, load_trip, project
from pathsnap.models.ensemble import (BASE_MODELS, TRAIN_SUBSAMPLE, CandidateEnsemble, hgb_is_tuned,
                                      hgb_params, logit, make_base_model, make_meta_learner)
from knn_sequence_metrics import route_metrics

CACHE = os.path.join(REPO, "data", "processed", "candidate_features")
OUT = os.path.join(REPO, "results", "ensemble")
VAL_EVERY = 3
ENSEMBLES = ["soft_vote", "stacking"]
ALL = BASE_MODELS + ENSEMBLES
LABELS = {"logistic_regression": "Logistic Regression", "knn_tuned": "KNN (Optuna-tuned)",
          "decision_tree": "Decision Tree", "hist_gradient_boosting": "Hist. Gradient Boosting",
          "soft_vote": "Ensemble: soft vote", "stacking": "Ensemble: stacking"}
METRICS = ["top1_accuracy", "macro_f1", "accuracy", "precision", "recall", "log_loss",
           "candidate_miss_rate"]


def data_root():
    if os.environ.get("PATHSNAP_DATA_ROOT"):
        return os.environ["PATHSNAP_DATA_ROOT"]
    if os.environ.get("PATHSNAP_TRAIN_ROOT"):
        return os.path.dirname(os.path.normpath(os.environ["PATHSNAP_TRAIN_ROOT"]))
    sys.exit("Set PATHSNAP_DATA_ROOT to the folder that contains train/ and student_test/")


def subset(tr, every):
    """Keep every n-th observation with all of its candidate rows (same rule as pathsnap.models.knn)."""
    sel = tr["obs"] % every == 0
    n = len(tr["true_arc"])
    return dict(tr, X=tr["X"][sel], y=tr["y"][sel], obs=tr["obs"][sel], cand=tr["cand"][sel],
                row_labelled=tr["row_labelled"][sel],
                true_arc=np.where(np.arange(n) % every == 0, tr["true_arc"], -1))


def scores(trips, probs, tables=None):
    r = K.evaluate(None, trips, tables, probs=probs)
    r.pop("inference_time_s")
    return r


def sequence_metrics(root, trips, probs):
    rows = []
    for tr in trips:
        tid = tr["trip"]
        _, nodes, arcs, track, route = load_trip(os.path.join(root, "student_test", tid))
        P = project(nodes[:, 0], nodes[:, 1], track[:, 0].mean(), track[:, 1].mean())
        arc_len = np.hypot(*(P[arcs[:, 1]] - P[arcs[:, 0]]).T)
        p, obs, cand = probs[tid], tr["obs"], tr["cand"]
        order = np.lexsort((-p, obs))
        first = np.r_[True, obs[order][1:] != obs[order][:-1]]
        seq = cand[order][first]
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
        prec, rec, f1, rmf = route_metrics(seq, route, arc_len)
        rows.append(dict(trip=tid, route_precision=prec, route_recall=rec, route_f1=f1,
                         route_mismatch_fraction=rmf))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "trip"}


def main():
    root = data_root()
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(K.SEED)
    K.THRESHOLD = 0.5                    # one row-level threshold for every model; Top-1 ignores it
    cache_all(root, CACHE)
    train, test = load_cache(CACHE, "train"), load_cache(CACHE, "student_test")
    for tr in train + test:
        tr["trip"] = str(tr["trip"])
    if (len(train), len(test)) != (80, 10):
        raise ValueError(f"expected 80 train and 10 student-test trips, got {len(train)}/{len(test)}")
    if hgb_is_tuned():
        LABELS["hist_gradient_boosting"] = "Hist. Gradient Boosting (Optuna-tuned)"
    print("boosting settings:", hgb_params(), flush=True)
    tables = {tr["trip"]: K.arc_table(root, "student_test", tr["trip"]) for tr in test}

    # ------------------------------------------------------------------ 5-fold grouped CV
    idx = np.arange(len(train))
    folds = list(GroupKFold(n_splits=5).split(idx, groups=idx))
    val = [subset(tr, VAL_EVERY) for tr in train]
    oof = {name: {} for name in BASE_MODELS}            # trip -> P(true arc) on its validation rows
    oof_full = {name: {} for name in ALL}               # trip -> P(true arc) on every row (HMM tuning)
    keep = {tr["trip"]: tr["obs"] % VAL_EVERY == 0 for tr in train}
    fold_of = {}
    cv = {name: [] for name in ALL}
    fit_s = {name: [] for name in BASE_MODELS}
    for k, (tr_i, va_i) in enumerate(folds):
        fold_trips = [val[i] for i in va_i]
        for i in va_i:
            fold_of[train[i]["trip"]] = k
        for name in BASE_MODELS:
            X, y, _ = K.stack([train[i] for i in tr_i], TRAIN_SUBSAMPLE[name])
            model = make_base_model(name)
            s = time.perf_counter(); model.fit(X, y); fit_s[name].append(time.perf_counter() - s)
            for i in va_i:
                trip = train[i]["trip"]
                oof_full[name][trip] = model.predict_proba(train[i]["X"])[:, 1]
                oof[name][trip] = oof_full[name][trip][keep[trip]]
            cv[name].append(scores(fold_trips, oof[name]))
        for tr in fold_trips:
            oof_full["soft_vote"][tr["trip"]] = np.mean([oof_full[n][tr["trip"]] for n in BASE_MODELS], axis=0)
        vote = {tr["trip"]: oof_full["soft_vote"][tr["trip"]][keep[tr["trip"]]] for tr in fold_trips}
        cv["soft_vote"].append(scores(fold_trips, vote))
        print(f"fold {k}: " + "  ".join(f"{n}={cv[n][-1]['top1_accuracy']:.4f}"
                                        for n in BASE_MODELS + ["soft_vote"]), flush=True)

    # stacking: meta-learner on OOF probabilities, scored with the same trip folds
    def meta_xy(trips):
        X = np.vstack([logit(np.column_stack([oof[n][tr["trip"]] for n in BASE_MODELS]))[tr["row_labelled"]]
                       for tr in trips])
        y = np.concatenate([tr["y"][tr["row_labelled"]] for tr in trips])
        return X, y

    def meta_probs(meta, tr, base_cols):
        return meta.predict_proba(logit(base_cols))[:, 1]

    for k, (tr_i, va_i) in enumerate(folds):
        meta = make_meta_learner().fit(*meta_xy([val[i] for i in tr_i]))
        for i in va_i:
            trip = train[i]["trip"]
            oof_full["stacking"][trip] = meta_probs(meta, train[i], np.column_stack(
                [oof_full[n][trip] for n in BASE_MODELS]))
        stacked = {val[i]["trip"]: oof_full["stacking"][val[i]["trip"]][keep[val[i]["trip"]]] for i in va_i}
        cv["stacking"].append(scores([val[i] for i in va_i], stacked))
    print("stacking folds: " + " ".join(f"{f['top1_accuracy']:.4f}" for f in cv["stacking"]), flush=True)

    cv_summary = {name: {m: dict(mean=float(np.mean([f[m] for f in cv[name]])),
                                 std=float(np.std([f[m] for f in cv[name]]))) for m in METRICS}
                  for name in ALL}
    chosen = max(ENSEMBLES, key=lambda n: cv_summary[n]["top1_accuracy"]["mean"])

    # base-model agreement on OOF rows: how often two models pick the same Top-1 arc
    def top1_arcs(trip_probs):
        out = []
        for tr in val:
            p, obs, cand = trip_probs[tr["trip"]], tr["obs"], tr["cand"]
            order = np.lexsort((-p, obs))
            out.append(cand[order][np.r_[True, obs[order][1:] != obs[order][:-1]]])
        return np.concatenate(out)
    picks = {n: top1_arcs(oof[n]) for n in BASE_MODELS}
    agreement = {a: {b: float(np.mean(picks[a] == picks[b])) for b in BASE_MODELS} for a in BASE_MODELS}

    # ------------------------------------------------------------------ refit on all 80 trips
    fitted, final_fit_s = {}, {}
    for name in BASE_MODELS:
        X, y, _ = K.stack(train, TRAIN_SUBSAMPLE[name])
        model = make_base_model(name)
        s = time.perf_counter(); model.fit(X, y); final_fit_s[name] = time.perf_counter() - s
        fitted[name] = model
        print(f"refit {name}: {len(y)} rows, {final_fit_s[name]:.1f}s", flush=True)
    meta = make_meta_learner().fit(*meta_xy(val))
    ensembles = {"soft_vote": CandidateEnsemble(fitted, "soft_vote"),
                 "stacking": CandidateEnsemble(fitted, "stacking", meta)}

    # ------------------------------------------------------------------ student test
    test_probs = {name: {} for name in ALL}
    infer_s = {name: 0.0 for name in BASE_MODELS}
    for tr in test:
        for name in BASE_MODELS:
            s = time.perf_counter()
            test_probs[name][tr["trip"]] = fitted[name].predict_proba(tr["X"])[:, 1]
            infer_s[name] += time.perf_counter() - s
        base = np.column_stack([test_probs[n][tr["trip"]] for n in BASE_MODELS])
        for name in ENSEMBLES:
            test_probs[name][tr["trip"]] = ensembles[name].combine(base)
    infer_s["soft_vote"] = infer_s["stacking"] = sum(infer_s[n] for n in BASE_MODELS)
    student = {}
    for name in ALL:
        r = scores(test, test_probs[name], tables)
        r.update(sequence_metrics(root, test, test_probs[name]))
        r["inference_time_s"] = infer_s[name]
        r["fit_time_s"] = (final_fit_s[name] if name in BASE_MODELS
                           else sum(final_fit_s.values()))
        r["per_trip_top1"] = {tr["trip"]: scores([tr], test_probs[name])["top1_accuracy"]
                              for tr in test}
        student[name] = r
        np.savez_compressed(os.path.join(OUT, f"{name}_student_test_probs.npz"), **test_probs[name])
        np.savez_compressed(os.path.join(OUT, f"{name}_train_oof_probs.npz"), **oof_full[name])

    # ------------------------------------------------------------------ save
    results = dict(
        features=FEATURES, seed=K.SEED, hgb_tuned=hgb_is_tuned(), hgb_params=hgb_params(), threshold=K.THRESHOLD, train_subsample_every=TRAIN_SUBSAMPLE,
        cv_protocol=f"5-fold GroupKFold by trip on 80 training trips; validation every {VAL_EVERY}rd observation",
        stacking=dict(meta_learner="LogisticRegression(C=1.0) on logit(base probabilities)",
                      trained_on="out-of-fold base probabilities, labelled validation rows",
                      coefficients=dict(zip(BASE_MODELS, meta.coef_[0].tolist())),
                      intercept=float(meta.intercept_[0])),
        chosen_ensemble=chosen, chosen_by="highest mean CV Top-1 (student test not used)",
        cv_folds=cv, cv_summary=cv_summary, cv_fit_time_s={n: float(np.mean(v)) for n, v in fit_s.items()},
        oof_top1_agreement=agreement, student_test=student)
    with open(os.path.join(OUT, "all_models_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    # the stacking object holds every fitted base model plus the meta-learner -> scripts/export_model.py
    joblib.dump(ensembles["stacking"], os.path.join(OUT, "trained_models.joblib"))

    cols = ["model", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_log_loss_mean",
            "test_top1", "test_macro_f1", "test_precision", "test_recall", "test_log_loss",
            "test_candidate_miss_rate", "test_route_f1", "test_route_mismatch_fraction",
            "test_route_continuity", "fit_time_s", "test_inference_time_s"]
    with open(os.path.join(OUT, "model_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for n in ALL:
            c, t = cv_summary[n], student[n]
            w.writerow([LABELS[n], c["top1_accuracy"]["mean"], c["top1_accuracy"]["std"],
                        c["macro_f1"]["mean"], c["log_loss"]["mean"], t["top1_accuracy"],
                        t["macro_f1"], t["precision"], t["recall"], t["log_loss"],
                        t["candidate_miss_rate"], t["route_f1"], t["route_mismatch_fraction"],
                        t["route_continuity"], t["fit_time_s"], t["inference_time_s"]])

    print(f"\n{'model':26s} {'CV top1':>15s} {'test top1':>9s} {'test F1':>8s} {'logloss':>8s} "
          f"{'routeF1':>8s} {'contin.':>8s}")
    for n in ALL:
        c, t = cv_summary[n]["top1_accuracy"], student[n]
        print(f"{LABELS[n]:26s} {c['mean']:.4f} ± {c['std']:.4f} {t['top1_accuracy']:9.4f} "
              f"{t['macro_f1']:8.4f} {t['log_loss']:8.4f} {t['route_f1']:8.4f} {t['route_continuity']:8.4f}")
    print(f"\nchosen ensemble: {chosen}; stacking weights: {results['stacking']['coefficients']}")


if __name__ == "__main__":
    main()
