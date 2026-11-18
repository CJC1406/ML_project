"""Report plots for the KNN model, generated only from results/knn_results.json (actual results)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
r = json.load(open(os.path.join(res_dir, "knn_results.json")))

# 1. CV macro F1 / Top-1 per fold
folds = r["cv_folds"]
x = np.arange(len(folds))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(x - 0.2, [f["macro_f1"] for f in folds], 0.4, label="Macro F1 (candidate rows)")
ax.bar(x + 0.2, [f["top1_accuracy"] for f in folds], 0.4, label="Top-1 accuracy (observations)")
ax.set_xticks(x, [f"Fold {f['fold']}" for f in folds]); ax.set_ylim(0.5, 1.0)
ax.set_title("KNN: 5-fold GroupKFold on 80 training trips"); ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(res_dir, "knn_cv_folds.png"), dpi=150)

# 2. Confusion matrix on student testing (threshold 0.5)
cm = np.array(r["student_test"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["pred: not true arc", "pred: true arc"], fontsize=8)
ax.set_yticks([0, 1], ["actual: not true arc", "actual: true arc"], fontsize=8)
ax.set_title("KNN confusion matrix (student test)")
fig.tight_layout(); fig.savefig(os.path.join(res_dir, "knn_confusion_matrix.png"), dpi=150)

# 3. Top-1 per student-testing trip
pt = r["student_test_per_trip"]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar([p["trip"][-2:] for p in pt], [p["top1_accuracy"] for p in pt])
ax.set_ylim(0.5, 1.0); ax.set_xlabel("student-test trip id"); ax.set_ylabel("Top-1 accuracy")
ax.set_title("KNN Top-1 accuracy per student-test trip")
fig.tight_layout(); fig.savefig(os.path.join(res_dir, "knn_student_test_per_trip.png"), dpi=150)
print("saved plots to", res_dir)
