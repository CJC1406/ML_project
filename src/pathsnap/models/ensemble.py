"""All PathSnap candidate scorers in one place, plus ensembles built from them.

Every model answers the same question for a (GPS observation, candidate arc) row: P(candidate is the
true arc). They all read the same 15 features from ``pathsnap.features.candidate_features`` so their
probabilities can be compared and combined row by row.

Base models (configurations as chosen by their owners; see docs/report/ensemble.md):
  * logistic_regression     StandardScaler + LogisticRegression(class_weight="balanced")
  * knn_tuned               Optuna-tuned KNN (k=31, distance weights, feature-group multipliers)
  * decision_tree           max_depth=12, min_samples_leaf=30
  * hist_gradient_boosting  Optuna-tuned (results/hgb_tuned/); falls back to 150 iterations,
                            learning rate 0.08, 31 leaves when no tuning result exists

Ensembles:
  * soft_vote   unweighted mean of the base probabilities
  * stacking    logistic-regression meta-learner on the logits of out-of-fold base probabilities
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pathsnap.features.candidate_features import FEATURES
from pathsnap.models.estimators.logistic_regression import build_logistic_regression
from pathsnap.models.knn import scale_cols

SEED = 42
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KNN_BEST = os.path.join(REPO, "results", "knn_tuned", "optuna_best.json")
HGB_BEST = os.path.join(REPO, "results", "hgb_tuned", "optuna_best.json")
HGB_UNTUNED = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31)
KNN_GROUPS = {
    "w_dist": ["dist", "dist_rank", "dist_minus_min", "win_dist"],
    "w_head": ["cos_heading", "heading_ok"],
    "w_cont": ["cont_prev", "cont_next"],
    "w_ctx": ["speed", "dt_prev", "arc_len", "proj_t", "n_cand", "junction_deg", "has_reverse"],
}
BASE_MODELS = ["logistic_regression", "knn_tuned", "decision_tree", "hist_gradient_boosting"]
# Fit on every n-th GPS observation (all of its candidate rows). KNN stores and scans its whole
# training set, so it keeps the subsample its Optuna study selected; the other models use every row.
TRAIN_SUBSAMPLE = {"logistic_regression": 1, "knn_tuned": 5, "decision_tree": 1,
                   "hist_gradient_boosting": 1}


def _knn_tuned():
    best = json.load(open(KNN_BEST))["best"]["params"]
    w = np.ones(len(FEATURES))
    for group, cols in KNN_GROUPS.items():
        for c in cols:
            w[FEATURES.index(c)] = best[group]
    return make_pipeline(
        StandardScaler(),
        FunctionTransformer(scale_cols, kw_args=dict(w=w)),
        KNeighborsClassifier(n_neighbors=best["n_neighbors"], weights=best["weights"],
                             p=best["p"], n_jobs=-1),
    )


def make_base_model(name: str):
    if name == "logistic_regression":
        return build_logistic_regression(random_state=SEED)
    if name == "knn_tuned":
        return _knn_tuned()
    if name == "decision_tree":
        return make_pipeline(SimpleImputer(strategy="median"),
                             DecisionTreeClassifier(max_depth=12, min_samples_leaf=30,
                                                    random_state=SEED))
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**hgb_params(), early_stopping=False, random_state=SEED)
    raise KeyError(name)


def hgb_params():
    """Optuna-tuned boosting settings when results/hgb_tuned/optuna_best.json exists, else the originals."""
    if not os.path.isfile(HGB_BEST):
        return dict(HGB_UNTUNED)
    p = dict(json.load(open(HGB_BEST))["best"]["params"])
    depth = p.pop("max_depth", None) if p.pop("limit_depth", False) else None
    return dict(p, max_depth=depth)


def hgb_is_tuned():
    return os.path.isfile(HGB_BEST)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_meta_learner():
    return LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)


class CandidateEnsemble:
    """Combines fitted base models. Exposes ``predict_proba`` and ``classes_`` like a classifier."""

    classes_ = np.array([0, 1])

    def __init__(self, base_models: dict, method: str = "soft_vote", meta=None):
        if method not in ("soft_vote", "stacking"):
            raise ValueError(method)
        if method == "stacking" and meta is None:
            raise ValueError("stacking needs a fitted meta-learner")
        self.base_models, self.method, self.meta = base_models, method, meta
        self.names = list(base_models)

    def base_probabilities(self, X):
        return np.column_stack([self.base_models[n].predict_proba(X)[:, 1] for n in self.names])

    def combine(self, base_probs):
        if self.method == "soft_vote":
            return base_probs.mean(axis=1)
        return self.meta.predict_proba(logit(base_probs))[:, 1]

    def predict_proba(self, X):
        p = self.combine(self.base_probabilities(X))
        return np.column_stack([1 - p, p])
