"""Optuna (Bayesian, TPE) tuning of Histogram Gradient Boosting, the best single candidate model.

Why this model: in results/ensemble/ it had the highest CV and student-test Top-1 of the four base
models, and the stacking ensemble gives it most of its weight.

Search space
  learning_rate      0.01 .. 0.3   (log)
  max_iter           100 .. 800
  max_leaf_nodes     8 .. 255      (log)
  min_samples_leaf   10 .. 1000    (log)
  l2_regularization  1e-4 .. 10    (log)
  max_features       0.4 .. 1.0    (share of features each split may use)
  max_depth          None or 3 .. 16
Objective: mean Top-1 accuracy over 5-fold GroupKFold by trip on the 80 training trips (the same folds
and the same every-3rd-observation validation rows as scripts/ensemble/compare_all_models.py).
A median pruner stops a trial after any fold where it is below the median of earlier trials.
The untuned configuration is enqueued as trial 0. Student-test and evaluator trips are never read.
TPE seed 42. The chosen configuration is refit on all 80 training trips and scored once on student test.

Usage (repo root): python scripts/hgb/hgb_optuna.py [n_trials]     Outputs: results/hgb_tuned/
"""
import json, os, sys, time
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "scripts", "ensemble"))
sys.path.insert(0, os.path.join(REPO, "scripts", "knn"))
import pathsnap.models.knn as K
from pathsnap.features.candidate_features import load_cache
from compare_all_models import CACHE, VAL_EVERY, data_root, scores, sequence_metrics, subset

OUT = os.path.join(REPO, "results", "hgb_tuned")
SEED = 42
BASELINE = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, min_samples_leaf=20,
                l2_regularization=1e-4, max_features=1.0, limit_depth=False)


def build(p):
    return HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        max_features=p["max_features"], max_depth=p.get("max_depth") if p["limit_depth"] else None,
        early_stopping=False, random_state=SEED)


def main(n_trials=40):
    os.makedirs(OUT, exist_ok=True)
    K.THRESHOLD = 0.5
    train = load_cache(CACHE, "train")
    for tr in train:
        tr["trip"] = str(tr["trip"])
    idx = np.arange(len(train))
    folds = []
    for tr_i, va_i in GroupKFold(n_splits=5).split(idx, groups=idx):
        X, y, _ = K.stack([train[i] for i in tr_i], 1)
        folds.append((X, y, [subset(train[i], VAL_EVERY) for i in va_i]))
    print(f"prepared {len(folds)} folds", flush=True)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 1000, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            limit_depth=trial.suggest_categorical("limit_depth", [False, True]))
        if p["limit_depth"]:
            p["max_depth"] = trial.suggest_int("max_depth", 3, 16)
        top1, ll, f1 = [], [], []
        s = time.perf_counter()
        for k, (X, y, val) in enumerate(folds):
            m = build(p).fit(X, y)
            r = scores(val, {tr["trip"]: m.predict_proba(tr["X"])[:, 1] for tr in val})
            top1.append(r["top1_accuracy"]); ll.append(r["log_loss"]); f1.append(r["macro_f1"])
            trial.report(float(np.mean(top1)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("top1_std", float(np.std(top1)))
        trial.set_user_attr("log_loss", float(np.mean(ll)))
        trial.set_user_attr("macro_f1", float(np.mean(f1)))
        trial.set_user_attr("seconds", time.perf_counter() - s)
        print(f"trial {trial.number}: top1={np.mean(top1):.4f} logloss={np.mean(ll):.4f} "
              f"({time.perf_counter()-s:.0f}s) {p}", flush=True)
        return float(np.mean(top1))

    study = optuna.create_study(direction="maximize", study_name="hgb_top1",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.enqueue_trial(BASELINE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    minutes = (time.time() - start) / 60
    study.trials_dataframe().to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)

    base = study.trials[0]
    best = study.best_trial
    try:
        importance = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as e:                      # importance needs >1 complete trial
        importance = {"error": str(e)}
    summary = dict(
        model="Histogram Gradient Boosting", n_trials=len(study.trials),
        n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        protocol=f"5-fold GroupKFold by trip, validation every {VAL_EVERY}rd observation, TPE seed {SEED}, median pruner",
        objective="mean CV Top-1 accuracy (maximise)", minutes=minutes,
        baseline=dict(params=BASELINE, cv_top1=base.value, **base.user_attrs),
        best=dict(trial=best.number, params=best.params, cv_top1=best.value, **best.user_attrs),
        param_importance=importance)

    # refit untuned and tuned on all 80 trips, score once on student test (reporting only)
    root = data_root()
    test = load_cache(CACHE, "student_test")
    for tr in test:
        tr["trip"] = str(tr["trip"])
    tables = {tr["trip"]: K.arc_table(root, "student_test", tr["trip"]) for tr in test}
    X, y, _ = K.stack(train, 1)
    for label, params in (("untuned", BASELINE), ("tuned", best.params)):
        p = dict(params)
        p.setdefault("limit_depth", False)
        s = time.perf_counter(); m = build(p).fit(X, y); fit_s = time.perf_counter() - s
        s = time.perf_counter(); probs = {tr["trip"]: m.predict_proba(tr["X"])[:, 1] for tr in test}
        inf_s = time.perf_counter() - s
        r = scores(test, probs, tables); r.update(sequence_metrics(root, test, probs))
        r.update(fit_time_s=fit_s, inference_time_s=inf_s)
        summary[f"student_test_{label}"] = r
        if label == "tuned":
            np.savez_compressed(os.path.join(OUT, "hgb_tuned_student_test_probs.npz"), **probs)
    with open(os.path.join(OUT, "optuna_best.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]   # pruned = partial folds
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter([t.number for t in done], [t.value * 100 for t in done], s=18, color="#2a78d6",
               label="completed trial")
    best_so_far = np.maximum.accumulate([t.value * 100 for t in done])
    ax.step([t.number for t in done], best_so_far, where="post", color="#eb6834", linewidth=2,
            label="best so far")
    ax.axhline(base.value * 100, color="#52514e", linestyle="--", linewidth=1, label="untuned (trial 0)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (%)")
    ax.set_title("Optuna study: Histogram Gradient Boosting", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "optuna_history.png"), dpi=150)

    t, u = summary["student_test_tuned"], summary["student_test_untuned"]
    print(f"\nCV Top-1 untuned {base.value:.4f} -> tuned {best.value:.4f} ({minutes:.1f} min)")
    print(f"student test Top-1 untuned {u['top1_accuracy']:.4f} -> tuned {t['top1_accuracy']:.4f}; "
          f"log loss {u['log_loss']:.4f} -> {t['log_loss']:.4f}")
    print("best params:", best.params)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
