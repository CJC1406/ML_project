"""Redraw results/hgb_tuned/optuna_history.png from optuna_trials.csv (completed trials only)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "hgb_tuned")
df = pd.read_csv(os.path.join(OUT, "optuna_trials.csv"))
done = df[df.state == "COMPLETE"].sort_values("number")
pruned = df[df.state == "PRUNED"]
fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.scatter(done.number, done.value * 100, s=18, color="#2a78d6", label="completed trial")
ax.step(done.number, np.maximum.accumulate(done.value * 100), where="post", color="#eb6834",
        linewidth=2, label="best so far")
ax.axhline(done.value.iloc[0] * 100, color="#52514e", linestyle="--", linewidth=1, label="untuned (trial 0)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (%)")
ax.set_title(f"Optuna study: Histogram Gradient Boosting ({len(pruned)} pruned trials not shown)",
             loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "optuna_history.png"), dpi=150)
print(done.value.max(), int(done.loc[done.value.idxmax(), "number"]))
