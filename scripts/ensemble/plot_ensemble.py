"""Plots for results/ensemble/all_models_results.json. Usage: python scripts/ensemble/plot_ensemble.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "ensemble")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CV_C, TEST_C = "#2a78d6", "#eb6834"
LABELS = {"logistic_regression": "Logistic Regression", "knn_tuned": "KNN (Optuna-tuned)",
          "decision_tree": "Decision Tree", "hist_gradient_boosting": "Hist. Gradient Boosting",
          "soft_vote": "Ensemble: soft vote", "stacking": "Ensemble: stacking"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


r = json.load(open(os.path.join(OUT, "all_models_results.json")))
names = list(r["cv_summary"])[::-1]
y = np.arange(len(names))
cv_m = [r["cv_summary"][n]["top1_accuracy"]["mean"] * 100 for n in names]
cv_s = [r["cv_summary"][n]["top1_accuracy"]["std"] * 100 for n in names]
te = [r["student_test"][n]["top1_accuracy"] * 100 for n in names]

fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
style(ax)
h = 0.36
ax.barh(y + h / 2, cv_m, h - 0.04, xerr=cv_s, color=CV_C, ecolor=INK2, capsize=2,
        error_kw=dict(linewidth=1), label="5-fold grouped CV (mean ± std)")
ax.barh(y - h / 2, te, h - 0.04, color=TEST_C, label="Student test (10 trips)")
for yi, v in zip(y, te):
    ax.text(v + 0.3, yi - h / 2, f"{v:.1f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, [LABELS[n] for n in names], color=INK)
lo = min(min(cv_m) - max(cv_s), min(te)) - 3
ax.set_xlim(max(0, lo), 100)
ax.set_xlabel("Top-1 accuracy (%)", color=INK2)
ax.set_title("Top-1 accuracy: base models and ensembles", color=INK, loc="left", fontsize=12)
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "all_models_top1.png"), dpi=150)

# Pairwise agreement of base models' Top-1 picks on out-of-fold rows (sequential, one hue)
base = list(r["oof_top1_agreement"])
A = np.array([[r["oof_top1_agreement"][a][b] for b in base] for a in base]) * 100
fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=SURFACE)
im = ax.imshow(A, cmap="Blues", vmin=min(80, A.min()), vmax=100)
short = [LABELS[n].replace(" (Optuna-tuned)", "").replace("Hist. Gradient Boosting", "HGB") for n in base]
ax.set_xticks(range(len(base)), short, rotation=30, ha="right", color=INK2, fontsize=8)
ax.set_yticks(range(len(base)), short, color=INK2, fontsize=8)
for i in range(len(base)):
    for j in range(len(base)):
        ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if A[i, j] > 95 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Same Top-1 arc chosen (% of CV observations)", color=INK, loc="left", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8).outline.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "base_model_agreement.png"), dpi=150)
print("saved plots to", os.path.abspath(OUT))
