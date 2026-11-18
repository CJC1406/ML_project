"""Tune and evaluate HMM/Viterbi route reconstruction on top of the candidate classifiers.

1. Tune  alpha (emission weight) and beta (transition scale, metres) with Optuna (TPE, seed 42) on the
         80 training trips, using the selected ensemble's OUT-OF-FOLD probabilities
         (results/ensemble/<ensemble>_train_oof_probs.npz), so no trip is decoded with probabilities from
         a model that was trained on it. Objective: observation-level Top-1 over labelled readings.
2. Test  decode the 10 student-test trips with every model's probabilities, classifier-only vs
         classifier + HMM/Viterbi, with the tuned parameters. Evaluator trips are never read.
Metrics  Top-1 (every labelled reading; no candidate counts as wrong), length-weighted route precision /
         recall / F1 / mismatch fraction of the reconstructed route, route continuity, chain breaks,
         and decode time.

Usage (repo root): python scripts/hmm/run_hmm.py [n_trials]          Outputs: results/hmm/
"""
import csv, json, os, sys, time
import numpy as np
import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "scripts", "knn"))
sys.path.insert(0, os.path.join(REPO, "scripts", "ensemble"))
from pathsnap.features.candidate_features import load_cache, load_trip
from pathsnap.matching.hmm import MAX_GAP_S, build_lattice, classifier_only, decode, reconstruct_route
from knn_sequence_metrics import route_metrics
from compare_all_models import ALL, CACHE, LABELS, data_root

ENS = os.path.join(REPO, "results", "ensemble")
OUT = os.path.join(REPO, "results", "hmm")
SEED = 42


def lattices(root, split, probs):
    out = []
    for tr in load_cache(CACHE, split):
        tid = str(tr["trip"])
        _, nodes, arcs, track, route = load_trip(os.path.join(root, split, tid))
        out.append(dict(trip=tid, lat=build_lattice(tr, probs[tid], nodes, arcs, track),
                        true_arc=tr["true_arc"], route=route))
    return out


def top1(items, preds):
    hit = n = 0
    for it, pred in zip(items, preds):
        lab = it["true_arc"] >= 0
        hit += int((pred[lab] == it["true_arc"][lab]).sum()); n += int(lab.sum())
    return hit / max(n, 1)


def evaluate(items, method, alpha=None, beta=None):
    preds, rows, breaks, t0 = [], [], 0, time.perf_counter()
    for it in items:
        if method == "hmm":
            pred, b = decode(it["lat"], alpha, beta); breaks += b
        else:
            pred = classifier_only(it["lat"])
        preds.append(pred)
    decode_s = time.perf_counter() - t0
    for it, pred in zip(items, preds):
        g = it["lat"].graph
        route, unjoined = reconstruct_route(g, pred)
        prec, rec, f1, rmf = route_metrics(route, it["route"], g.length)
        cont = float(np.mean(g.arcs[route[:-1], 1] == g.arcs[route[1:], 0])) if len(route) > 1 else 1.0
        lab = it["true_arc"] >= 0
        rows.append(dict(trip=it["trip"], top1=float(np.mean(pred[lab] == it["true_arc"][lab])),
                         route_precision=prec, route_recall=rec, route_f1=f1,
                         route_mismatch_fraction=rmf, route_continuity=cont, unjoined_gaps=unjoined,
                         predicted_route_arcs=int(len(route)), reference_route_arcs=int(len(it["route"]))))
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "trip"}
    summary.update(top1_micro=top1(items, preds), chain_breaks=breaks, decode_seconds=decode_s)
    return summary, rows, preds


def main(n_trials=30):
    root = data_root()
    os.makedirs(OUT, exist_ok=True)
    chosen = json.load(open(os.path.join(ENS, "all_models_results.json")))["chosen_ensemble"]
    oof = np.load(os.path.join(ENS, f"{chosen}_train_oof_probs.npz"))
    s = time.perf_counter()
    train = lattices(root, "train", oof)
    print(f"built {len(train)} training lattices in {time.perf_counter()-s:.0f}s", flush=True)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
        beta = trial.suggest_float("beta", 0.5, 200.0, log=True)
        score = top1(train, [decode(it["lat"], alpha, beta)[0] for it in train])
        print(f"trial {trial.number}: alpha={alpha:.3f} beta={beta:.2f} top1={score:.4f}", flush=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(alpha=1.0, beta=10.0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = time.perf_counter(); study.optimize(objective, n_trials=n_trials); tune_min = (time.perf_counter() - s) / 60
    study.trials_dataframe().to_csv(os.path.join(OUT, "hmm_optuna_trials.csv"), index=False)
    alpha, beta = study.best_params["alpha"], study.best_params["beta"]
    train_clf, _, _ = evaluate(train, "classifier")
    train_hmm, _, _ = evaluate(train, "hmm", alpha, beta)
    print(f"train (OOF {chosen}): classifier top1={train_clf['top1_micro']:.4f} -> hmm {train_hmm['top1_micro']:.4f}", flush=True)

    config = dict(alpha=alpha, beta=beta, max_gap_s=MAX_GAP_S, transition_limit="2 * gps distance + 200 m",
                  route_fill_limit_m=2000.0, tuned_on=f"80 training trips, out-of-fold {chosen} probabilities",
                  objective="observation Top-1 over labelled readings", n_trials=n_trials,
                  tuning_minutes=tune_min, probability_model=chosen)
    json.dump(config, open(os.path.join(OUT, "hmm_config.json"), "w"), indent=2)

    results = dict(config=config, train_oof=dict(classifier_only=train_clf, hmm=train_hmm), student_test={})
    table = []
    for name in ALL:
        probs = np.load(os.path.join(ENS, f"{name}_student_test_probs.npz"))
        items = lattices(root, "student_test", probs)
        clf, clf_rows, _ = evaluate(items, "classifier")
        hmm, hmm_rows, preds = evaluate(items, "hmm", alpha, beta)
        results["student_test"][name] = dict(classifier_only=clf, hmm=hmm, per_trip_hmm=hmm_rows,
                                             per_trip_classifier_only=clf_rows)
        if name == chosen:
            np.savez_compressed(os.path.join(OUT, "student_test_hmm_predictions.npz"),
                                **{it["trip"]: p for it, p in zip(items, preds)})
        for method, r in (("classifier only", clf), ("+ HMM/Viterbi", hmm)):
            table.append([LABELS[name], method, r["top1_micro"], r["route_precision"], r["route_recall"],
                          r["route_f1"], r["route_mismatch_fraction"], r["route_continuity"], r["chain_breaks"],
                          r["decode_seconds"]])
        print(f"{LABELS[name]:28s} top1 {clf['top1_micro']:.4f} -> {hmm['top1_micro']:.4f}   "
              f"route F1 {clf['route_f1']:.3f} -> {hmm['route_f1']:.3f}   "
              f"mismatch {clf['route_mismatch_fraction']:.3f} -> {hmm['route_mismatch_fraction']:.3f}", flush=True)
    json.dump(results, open(os.path.join(OUT, "hmm_results.json"), "w"), indent=2, default=float)
    with open(os.path.join(OUT, "hmm_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "top1", "route_precision", "route_recall", "route_f1",
                    "route_mismatch_fraction", "route_continuity", "chain_breaks", "decode_seconds"])
        w.writerows(table)
    print(f"\nHMM parameters: alpha={alpha:.3f} beta={beta:.2f} (tuned {tune_min:.1f} min)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
