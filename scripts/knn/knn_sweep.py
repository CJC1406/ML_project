"""KNN sensitivity study: how k and the weighting scheme change train vs validation performance.

Purpose: evidence for the over/under-fitting question (report Q9) and justification of the chosen
configuration (k=25, distance weighting). This is a small manual grid, NOT the Bayesian optimisation
(Optuna is run later, only on the team's selected model).

Protocol: same 5-fold GroupKFold by trip on the 80 training trips and same features/subsampling as
train_knn.py. To keep run time practical on a laptop, validation is scored on every 12th GPS
observation (all of its candidates) and the training-fold score on every 40th observation.
"""
import json, os, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from pathsnap.features.candidate_features import load_cache
import pathsnap.models.knn as T

GRID = [(1, "distance"), (5, "distance"), (25, "distance"), (101, "distance"), (25, "uniform")]


def main(cache_dir, out_dir):
    train = load_cache(cache_dir, "train")
    for tr in train:
        tr["trip"] = str(tr["trip"])
    idx = np.arange(len(train))
    rows = []
    for k, w in GRID:
        T.CONFIG.update(n_neighbors=k, weights=w)
        fold_val, fold_tr = [], []
        s = time.perf_counter()
        for f, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(idx, groups=idx)):
            X, y, _ = T.stack([train[i] for i in tr_i], T.SUBSAMPLE)
            m = T.make_model(); m.fit(X, y)
            fold_val.append(T.evaluate(m, [train[i] for i in va_i], None, obs_every=12))
            # training-fold score: rows the model was fitted on (obs % SUBSAMPLE == 0) -> obs % 40 == 0
            fold_tr.append(T.evaluate(m, [train[i] for i in tr_i], None, obs_every=40))
        r = dict(k=k, weights=w, minutes=(time.perf_counter() - s) / 60)
        for name, fl in (("val", fold_val), ("train", fold_tr)):
            for met in ("macro_f1", "top1_accuracy", "log_loss"):
                v = [x[met] for x in fl]
                r[f"{name}_{met}_mean"] = float(np.mean(v)); r[f"{name}_{met}_std"] = float(np.std(v))
        rows.append(r)
        print(f"k={k:3d} {w:8s} train F1={r['train_macro_f1_mean']:.4f}  val F1={r['val_macro_f1_mean']:.4f}"
              f"±{r['val_macro_f1_std']:.4f}  val top1={r['val_top1_accuracy_mean']:.4f}  "
              f"val logloss={r['val_log_loss_mean']:.4f}  ({r['minutes']:.1f} min)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(os.path.join(out_dir, "knn_sweep.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dist = [r for r in rows if r["weights"] == "distance"]
    ks = [r["k"] for r in dist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, [r["train_macro_f1_mean"] for r in dist], "o-", label="training folds")
    ax[0].errorbar(ks, [r["val_macro_f1_mean"] for r in dist], yerr=[r["val_macro_f1_std"] for r in dist],
                   fmt="s-", capsize=3, label="validation folds")
    ax[0].set_xscale("log"); ax[0].set_xlabel("k (number of neighbours, log scale)"); ax[0].set_ylabel("Macro F1")
    ax[0].set_title("KNN (distance-weighted): train vs validation"); ax[0].legend()
    ax[1].plot(ks, [r["val_log_loss_mean"] for r in dist], "s-", color="C3")
    ax[1].set_xscale("log"); ax[1].set_xlabel("k (log scale)"); ax[1].set_ylabel("Validation log loss")
    ax[1].set_title("Probability quality vs k")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "knn_sweep.png"), dpi=150)
    json.dump(rows, open(os.path.join(out_dir, "knn_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
