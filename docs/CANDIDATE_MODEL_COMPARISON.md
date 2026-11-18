# PathSnap candidate model comparison

Both models were trained on the supplied 80 training trips and tested on the 10 separate student-test trips. A 64/16 trip-grouped holdout within training gave a validation check before refitting on all 80 trips. The evaluator partition was excluded.

## Results

| Method | Validation Top-1 | Student-test Top-1 | Top-1 across all aligned test readings | Test candidate F1 | Test candidate log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearest road | 54.57% | 55.77% | 55.68% | — | — |
| Decision Tree | 77.47% | 85.32% | 85.19% | 0.8444 | 0.1663 |
| Histogram Gradient Boosting | 77.86% | 85.44% | 85.31% | 0.8483 | 0.1413 |

Top-1 is micro-averaged across 31,096 validation readings or 14,036 test readings with a derived label present in the candidate list. The all-aligned column counts the 22 test readings whose aligned road was missing from the list as misses. Candidate F1 uses a probability threshold of 0.5. Candidate log loss evaluates the binary probability assigned to each row in scored candidate lists.

The test partition had 19,160 GPS readings, of which 14,058 received a derived label and 14,036 had that label in their candidate list. All test readings had at least one nearby road. On training there were 208,417 readings, 143,364 derived labels, and 142,911 labels covered by the candidate search.

Boosting ranked the derived reference first for 11,993 test readings versus 11,976 for Decision Tree. It improved test Top-1 by 0.12 percentage points and produced lower candidate log loss. Decision Tree fit in 6.15 seconds and scored the test batch in 0.011 seconds. Boosting fit in 8.49 seconds and scored the same batch in 0.362 seconds. Timings are from a single local run with thread pools limited to four threads and exclude preprocessing, candidate generation and alignment.

The Decision Tree assigned most of its impurity-based importance to distance above the nearest road (0.582) and heading difference (0.336). These values describe which variables its splits used and can shift when distance features are correlated.

## Method and limits

Candidate search uses a 50 m radius and keeps the nearest eight road arcs in the trip's projected coordinate system. Both models use the same six features: distance, heading difference, speed, time gap, candidate rank, and distance above the nearest road. Reference routes supply targets only.

Label derivation uses dynamic programming over nearby occurrences in the ordered reference route within 30 m. It allows repeated observations on one arc and forward skips, charges a cost for skipped arcs, and disallows transitions across breaks in the reference. Runs restart at missing reference candidates, gaps over 60 seconds, or infeasible transitions. A reading remains unlabelled when the best different-arc path is within five cost units. These are heuristic settings. Smoke checks cover ordering, ambiguity and breaks. Observation-level ground truth has not been independently audited. Results measure agreement with derived labels and the evaluation omits unlabelled readings. The experiment evaluates candidate scoring before route reconstruction.

The Decision Tree uses maximum depth 12, a minimum of 30 rows per leaf, and median imputation. Histogram Gradient Boosting uses 150 iterations, learning rate 0.08, up to 31 leaves per tree, and native missing-value handling. Both use seed 2026 and fixed settings. No hyperparameter search was run.

## Reproduce

From the repository root, pass the extracted dataset folder containing `train` and `student_test`.

```bash
python -m pip install -e ".[runtime]"
python scripts/compare_candidate_models.py "/path/to/Student folder" --out data/processed/candidate_comparison.json
```

The reference run used Python 3.13.5, NumPy 2.3.2, Shapely 2.1.2 (GEOS 3.13.1), PyProj 3.7.2 and scikit-learn 1.7.1. Shapely's `dwithin` query requires GEOS 3.10 or newer. See [the aggregate metrics](CANDIDATE_MODEL_RESULTS.json) for full precision.
