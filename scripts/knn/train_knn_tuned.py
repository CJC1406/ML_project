"""Retrain the KNN with the Optuna-selected configuration and evaluate it with the standard protocol.

Final configuration = best Optuna trial (results/knn_tuned/optuna_best.json)
                    + training subsample every 5th observation (subsample_threshold_experiment.json:
                      every 5th beat every 10th; every 3rd gave no further gain at higher cost)
                    + decision threshold 0.425 (chosen by nested CV on training folds only).
Same 5-fold GroupKFold by trip (validation every 3rd observation) and student-test evaluation as the
baseline, so the numbers are directly comparable with results/knn/.
Usage (repo root): python3 scripts/knn/train_knn_tuned.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, HERE)
import pathsnap.models.knn as T
import knn_optuna as O
from run_knn import DATA

OUT = os.path.join(REPO, "results", "knn_tuned")
best = json.load(open(os.path.join(OUT, "optuna_best.json")))["best"]["params"]
T.CONFIG.update(n_neighbors=best["n_neighbors"], weights=best["weights"], p=best["p"])
T.FEATURE_WEIGHTS = O.weight_vector(best).tolist()
T.SUBSAMPLE = 5
T.THRESHOLD = 0.425
T.MODEL_NAME = "KNN (Optuna-tuned)"
T.main(DATA, os.path.join(REPO, "data", "processed", "candidate_features"), OUT)
import knn_sequence_metrics as S
S.main(DATA, os.path.join(REPO, "data", "processed", "candidate_features"), OUT)
