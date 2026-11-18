"""Optuna (Bayesian / TPE) hyperparameter optimisation of the KNN candidate classifier.

Search space
  n_neighbors        5 .. 200 (log scale)
  weights            uniform | distance
  p                  1 (Manhattan) | 2 (Euclidean)
  feature-group multipliers applied after StandardScaler (feature-weighted distance), 0.1 .. 5 (log):
      w_dist  -> dist, dist_rank, dist_minus_min, win_dist
      w_head  -> cos_heading, heading_ok
      w_cont  -> cont_prev, cont_next
      w_ctx   -> speed, dt_prev, arc_len, proj_t, n_cand, junction_deg, has_reverse
  (multiplier 1 everywhere = plain KNN; a large multiplier makes that group dominate the distance,
   a small one makes the model almost ignore it -> soft feature selection)

Objective: mean validation macro F1 over 5-fold GroupKFold by trip on the 80 training trips (same folds
as the baseline). Scalers are fitted on each training fold only. Validation rows = every 20th GPS
observation (all its candidates) to keep each trial ~1 min; the final chosen configuration is then
re-evaluated with the standard protocol (every 3rd observation) by pathsnap.models.knn.main.
The untuned baseline is enqueued as trial 0 so it is compared under the identical protocol.
Student-test trips are never used during the search. Seed 42 (TPE sampler).

Usage (repo root):  python3 scripts/knn/knn_optuna.py [n_trials]
Outputs: results/knn_tuned/  (optuna_trials.csv, optuna_best.json, plots, tuned model results)
"""
import json, os, sys, time
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from pathsnap.features.candidate_features import FEATURES, load_cache
import pathsnap.models.knn as T

GROUPS = {
    "w_dist": ["dist", "dist_rank", "dist_minus_min", "win_dist"],
    "w_head": ["cos_heading", "heading_ok"],
    "w_cont": ["cont_prev", "cont_next"],
    "w_ctx": ["speed", "dt_prev", "arc_len", "proj_t", "n_cand", "junction_deg", "has_reverse"],
}
VAL_EVERY = 20
SEED = 42


def weight_vector(params):
    w = np.ones(len(FEATURES))
    for g, cols in GROUPS.items():
        for c in cols:
            w[FEATURES.index(c)] = params.get(g, 1.0)
    return w


class _Scaled:
    """Wraps a fitted KNN so evaluate() can pass pre-standardised rows; applies the feature weights."""
    def __init__(self, knn, w):
        self.knn, self.w = knn, w

    def predict_proba(self, X):
        return self.knn.predict_proba(X * self.w)


def prepare_folds(train):
    idx = np.arange(len(train))
    folds = []
    for tr_i, va_i in GroupKFold(n_splits=5).split(idx, groups=idx):
        X, y, _ = T.stack([train[i] for i in tr_i], T.SUBSAMPLE)
        sc = StandardScaler().fit(X)
        val = []
        for i in va_i:
            tr = train[i]
            sel = tr["obs"] % VAL_EVERY == 0
            n = len(tr["true_arc"])
            val.append(dict(tr, X=sc.transform(tr["X"][sel]), y=tr["y"][sel], obs=tr["obs"][sel],
                            cand=tr["cand"][sel], row_labelled=tr["row_labelled"][sel],
                            true_arc=np.where(np.arange(n) % VAL_EVERY == 0, tr["true_arc"], -1)))
        folds.append((sc.transform(X), y, val))
    return folds


def main(n_trials=40):
    out = os.path.join(REPO, "results", "knn_tuned")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(REPO, "data", "processed", "candidate_features")
    train = load_cache(cache, "train")
    for tr in train:
        tr["trip"] = str(tr["trip"])
    folds = prepare_folds(train)

    def objective(trial):
        k = trial.suggest_int("n_neighbors", 5, 200, log=True)
        wts = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_categorical("p", [1, 2])
        params = {g: trial.suggest_float(g, 0.1, 5.0, log=True) for g in GROUPS}
        w = weight_vector(params)
        f1s, top1s, lls = [], [], []
        for f, (Xtr, ytr, val) in enumerate(folds):
            knn = KNeighborsClassifier(n_neighbors=k, weights=wts, p=p, n_jobs=-1).fit(Xtr * w, ytr)
            r = T.evaluate(_Scaled(knn, w), val, None, obs_every=1)
            f1s.append(r["macro_f1"]); top1s.append(r["top1_accuracy"]); lls.append(r["log_loss"])
            trial.report(float(np.mean(f1s)), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("f1_std", float(np.std(f1s)))
        trial.set_user_attr("top1", float(np.mean(top1s)))
        trial.set_user_attr("log_loss", float(np.mean(lls)))
        return float(np.mean(f1s))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="knn",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.enqueue_trial(dict(n_neighbors=25, weights="distance", p=2,
                             w_dist=1.0, w_head=1.0, w_cont=1.0, w_ctx=1.0))   # = untuned baseline

    t0 = time.perf_counter()

    def log(st, tr):
        v = "pruned" if tr.state != optuna.trial.TrialState.COMPLETE else f"F1={tr.value:.4f} top1={tr.user_attrs.get('top1', 0):.4f}"
        print(f"trial {tr.number:3d} {v:32s} best={st.best_value:.4f}  {tr.params}  "
              f"[{(time.perf_counter() - t0) / 60:.1f} min]", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[log])

    df = study.trials_dataframe()
    df.to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
    base = study.trials[0]
    best = study.best_trial
    summary = dict(n_trials=len(study.trials),
                   n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                   protocol=f"5-fold GroupKFold by trip, validation every {VAL_EVERY}th observation, TPE seed {SEED}",
                   baseline=dict(params=base.params, cv_macro_f1=base.value, **base.user_attrs),
                   best=dict(trial=best.number, params=best.params, cv_macro_f1=best.value, **best.user_attrs),
                   minutes=(time.perf_counter() - t0) / 60)
    try:
        summary["param_importance"] = {k: float(v) for k, v in
                                       optuna.importance.get_param_importances(study).items()}
    except Exception as e:                      # needs enough completed trials
        summary["param_importance"] = str(e)
    json.dump(summary, open(os.path.join(out, "optuna_best.json"), "w"), indent=2)

    # plots: optimisation history + parameter importance
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].scatter([t.number for t in done], [t.value for t in done], s=14, label="trial (completed)")
    ax[0].plot([t.number for t in done], np.maximum.accumulate([t.value for t in done]), "r-", label="best so far")
    ax[0].axhline(base.value, color="gray", ls="--", label="baseline (k=25)")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro F1"); ax[0].set_title("Optuna optimisation history")
    ax[0].legend(fontsize=8)
    if isinstance(summary["param_importance"], dict):
        imp = summary["param_importance"]
        ax[1].barh(list(imp)[::-1], list(imp.values())[::-1])
        ax[1].set_xlabel("importance (fANOVA)"); ax[1].set_title("Hyperparameter importance")
    fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_history.png"), dpi=150)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
