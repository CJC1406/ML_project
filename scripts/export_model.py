"""Export the trained models and HMM settings as a versioned bundle for the FastAPI backend.

Inputs  results/ensemble/trained_models.joblib   (scripts/ensemble/compare_all_models.py)
        results/ensemble/all_models_results.json
        results/hmm/hmm_config.json, hmm_results.json   (scripts/hmm/run_hmm.py)
Output  models/pathsnap_v1/   (see pathsnap.serving.bundle for the layout)
Usage   python scripts/export_model.py [out_dir]
"""
import datetime, json, os, platform, subprocess, sys
from pathlib import Path

import joblib, numpy, scipy, sklearn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pathsnap.features import candidate_features as CF
from pathsnap.models.ensemble import TRAIN_SUBSAMPLE, hgb_params
from pathsnap.serving.bundle import load_bundle, save_bundle

LABELS = {"logistic_regression": "Logistic Regression", "knn_tuned": "KNN (Optuna-tuned)",
          "decision_tree": "Decision Tree", "hist_gradient_boosting": "Hist. Gradient Boosting (Optuna-tuned)",
          "soft_vote": "Ensemble: soft vote", "stacking": "Ensemble: stacking"}


def main(out):
    ens = joblib.load(REPO / "results/ensemble/trained_models.joblib")
    res = json.loads((REPO / "results/ensemble/all_models_results.json").read_text())
    hmm_cfg = json.loads((REPO / "results/hmm/hmm_config.json").read_text())
    hmm_res = json.loads((REPO / "results/hmm/hmm_results.json").read_text())
    knn_best = json.loads((REPO / "results/knn_tuned/optuna_best.json").read_text())["best"]["params"]
    if not res.get("hgb_tuned"):
        raise SystemExit("results/ensemble was produced with untuned boosting; re-run compare_all_models.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    metrics = []
    for name, label in LABELS.items():
        t, h = res["student_test"][name], hmm_res["student_test"][name]
        metrics.append(dict(
            model=name, label=label, accuracy=t["accuracy"], precision=t["precision"], recall=t["recall"],
            macro_f1=t["macro_f1"], log_loss=t["log_loss"], top1_classifier_only=h["classifier_only"]["top1_micro"],
            top1_hmm=h["hmm"]["top1_micro"], route_f1_hmm=h["hmm"]["route_f1"],
            route_mismatch_hmm=h["hmm"]["route_mismatch_fraction"],
            cv_top1_mean=res["cv_summary"][name]["top1_accuracy"]["mean"],
            protocol="student-test trips (10), models refit on 80 training trips; row metrics at threshold 0.5"))
    extra = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit, default_model=res["chosen_ensemble"], model_labels=LABELS,
        environment=dict(python=platform.python_version(), numpy=numpy.__version__,
                         scipy=scipy.__version__, scikit_learn=sklearn.__version__),
        features=CF.FEATURES,
        candidates=dict(radius_m=CF.RADIUS_M, max_candidates=CF.MAX_CAND, densify_m=CF.DENSIFY_M,
                        smoothing_window=CF.WIN),
        configs=dict(hist_gradient_boosting=hgb_params(), knn_tuned=knn_best,
                     decision_tree=dict(max_depth=12, min_samples_leaf=30),
                     logistic_regression=dict(class_weight="balanced", max_iter=1000),
                     train_subsample_every=TRAIN_SUBSAMPLE, stacking=res["stacking"]),
        training_data=dict(train_trips=80, student_test_trips=10, evaluator_trips_used=False),
        hmm=hmm_cfg, metrics=metrics)
    manifest = save_bundle(Path(out), ens.base_models, ens.meta, hmm_cfg, extra)
    models, _, _ = load_bundle(Path(out))          # round trip with checksum verification
    sizes = {f: (Path(out) / f).stat().st_size / 1e6 for f in manifest["files"]}
    print(f"exported {list(models)} to {out}")
    for f, mb in sizes.items():
        print(f"  {f:40s} {mb:7.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(REPO / "models" / "pathsnap_v1"))
