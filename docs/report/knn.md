# Model: K-Nearest Neighbours (KNN)

All numbers come from `results/knn/` (produced by `scripts/knn/run_knn.py` and `scripts/knn/knn_sweep.py`, seed 42).

**Task.** For each GPS observation we generate up to 10 candidate arcs within 50 m. The model sees one row per (observation, candidate) with 15 features and predicts P(candidate is the true arc). Labels come from a monotone alignment of the track to the ground-truth route.

**Features (15):** dist, dist_rank, dist_minus_min, win_dist, cos_heading, heading_ok, speed, dt_prev, arc_len, proj_t, n_cand, junction_deg, has_reverse, cont_prev, cont_next.

## Q1. Why this algorithm suits PathSnap
A candidate is probably correct if it looks like correct candidates we have already seen: it is close to the GPS point, its direction matches the vehicle's heading, and it connects to neighbouring candidates. KNN uses exactly this rule. It finds the 25 most similar labelled rows from the training trips and lets them vote. It makes no assumption about the shape of the decision boundary, which helps here because the important interactions are non-linear (for example, a small distance only helps when the heading also matches). Training is effectively free (fit time 0.17 s), so it gives a strong, explainable baseline against the parametric and tree models.

## Q2. Strengths and limitations
**Strengths**
- Non-parametric: it captures non-linear interactions between features with no feature engineering beyond scaling.
- Easy to explain in the viva: "these 25 similar past candidates were mostly correct."
- Almost no training time; well-calibrated probabilities once k is large (log loss 0.139 on student test).

**Limitations**
- Slow at prediction time: every query searches the stored training set (83.8 s for 131,570 rows, the slowest of the families). To keep this manageable we store only every 10th observation.
- It needs feature scaling (StandardScaler); without it, features measured in metres or seconds would dominate the distance.
- All 15 features count equally in the distance, so weak features add noise.
- It scores each candidate on its own, so the predicted route can jump between parallel or reverse arcs (route continuity 0.807). This is why the HMM/Viterbi stage is needed.

## Q5. Important hyperparameters and initial values
| Hyperparameter | Value used | Notes |
|---|---|---|
| n_neighbors (k) | 25 | Initial value 25; sweep over {1, 5, 25, 101} confirmed it is best by CV macro F1 |
| weights | distance | Closer neighbours count more; uniform gives almost the same F1 (0.900 vs 0.901) |
| metric | Euclidean (Minkowski p=2) | On standardised features |
| algorithm | auto (KD/ball tree) | Affects speed only |
| Scaling | StandardScaler | Fitted on the training folds only |
| Training subsample | every 10th observation | Speed–accuracy trade-off; keeps all candidates of the chosen observations |

Sweep (5-fold GroupKFold by trip; `results/knn/knn_sweep.csv`, `knn_sweep.png`):

| k | weights | Train F1 | Val F1 | Val Top-1 | Val log loss |
|---|---|---|---|---|---|
| 1 | distance | 1.000 | 0.872 ± 0.029 | 0.797 | 0.916 |
| 5 | distance | 1.000 | 0.894 ± 0.030 | 0.826 | 0.302 |
| **25** | **distance** | 1.000 | **0.901 ± 0.029** | **0.843** | 0.167 |
| 101 | distance | 1.000 | 0.895 ± 0.028 | 0.840 | 0.148 |
| 25 | uniform | 0.909 | 0.900 ± 0.029 | 0.843 | 0.169 |

## Q6. Loss / optimisation objective
KNN has no loss function and no training optimisation: "fitting" only stores the scaled training rows. A prediction is the distance-weighted share of the k neighbours that are labelled "true arc". We evaluate those probabilities with **log loss** (binary cross-entropy) and choose k by cross-validated **macro F1**, so model selection is the only place an objective is optimised. Because KNN has no epochs, we report no training curve; the k-sweep plot plays that role.

## Q7. Student-test performance (10 trips, 131,570 candidate rows)
| Metric | Value |
|---|---|
| Accuracy | 0.956 |
| Precision (true-arc class) | 0.863 |
| Recall (true-arc class) | 0.828 |
| Macro F1 | 0.910 |
| Log loss | 0.139 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[109958, 2504], [3280, 15828]] |
| Candidate miss rate | 0.13 % |
| Route continuity | 0.807 |
| Fit time / inference time | 0.17 s / 83.8 s |

Sequence level, KNN alone (Top-1 arc per point, no HMM): route precision 0.848, recall 0.974, F1 0.904, Newson–Krumm mismatch 0.214 (`knn_sequence_metrics.json`). Recall is high and precision is lower: the true route is almost always found, but extra wrong arcs (reverse twins and parallel roads) get added.

## Q8. Top-1, macro F1 and log loss on student test
**Top-1 accuracy 0.853 · macro F1 0.910 · log loss 0.139.**

## Q9. Over/underfitting, with evidence
- **No meaningful overfitting at k = 25.** CV macro F1 0.901 ± 0.030 versus student-test 0.910, and CV Top-1 0.843 versus test 0.853. The test numbers are within one standard deviation of CV, so the model generalises to unseen trips.
- **k = 1 overfits:** training F1 1.000 vs validation 0.872, and log loss 0.916, because its probabilities are hard 0/1 votes and every mistake is confidently wrong.
- **Large k starts to underfit:** k = 101 drops validation F1 to 0.895 (the boundary smooths over), even though log loss improves slightly to 0.148.
- Note: with distance weighting the training F1 is always 1.000, because each training row is its own neighbour at distance 0. The fairer train-vs-validation comparison is uniform k = 25: train 0.909 vs validation 0.900, a gap of less than 0.01.
- Fold spread (0.88–0.94) comes from trip difficulty, not from model variance; for example, fold 2 contains trips with noisy ground truth.
