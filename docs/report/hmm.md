# HMM and Viterbi route reconstruction

Code: `src/pathsnap/matching/hmm.py` (model, decoder, route reconstruction) and `scripts/hmm/run_hmm.py` (tuning and evaluation). Tests: `tests/test_hmm.py`. Results: `results/hmm/`.

## Why a sequence model

The classifiers score each GPS reading on its own. When two candidates look alike, such as the two directions of a two-way road, a parallel service road or a junction, the best-scoring arc can flip from one reading to the next. That gives jumps the vehicle could not have made. The HMM adds the missing constraint: consecutive readings must be explained by a drive along the road network that is about as long as the distance between the two GPS fixes.

## Model

| Part | Definition |
| --- | --- |
| Hidden state at reading *i* | One of its candidate arcs (up to 10 within 50 m), the same set the classifier scored |
| Emission | log e_i(a) = α · log P̂(a), where P̂ is the classifier probability normalised over reading *i*'s candidates |
| Transition | log T(a→b) = −\|d_road(a,b) − d_gps(i,i+1)\| / β, following Newson and Krumm (2009) |
| d_road | Driving distance from the reading's projected position on arc *a* to the next reading's projected position on arc *b*. Staying on the same arc moves forward along it, and small backward GPS jitter costs nothing. Other moves use the shortest path from the end of *a* to the start of *b*. |
| Impossible moves | *b* not reachable within 2 · d_gps + 200 m, so log T = −∞. This covers disconnected components and wrong-way moves. |
| Chain restarts | When a reading has no candidates, the time gap exceeds 60 s, or no finite transition exists, Viterbi starts a new segment instead of forcing an impossible path. |
| Decoding | Viterbi in log space. For each state it keeps the best score and a back-pointer, then traces back from the best final state of each segment. |

Recurrence, with s_i(b) the best log score of any path ending in state *b* at reading *i*:

```
s_1(b) = log e_1(b)
s_i(b) = log e_i(b) + max_a [ s_{i-1}(a) + log T(a -> b) ]
ptr_i(b) = argmax_a [ s_{i-1}(a) + log T(a -> b) ]
```

Shortest paths are computed lazily with a distance-limited Dijkstra search from each candidate arc's end node, and the results are cached. Building the lattices for all 80 training trips takes 54 s. Decoding them all takes about 5 s.

**Route reconstruction.** The decoded arc per reading is collapsed into a sequence without repeats. Any two consecutive arcs that do not touch are joined by the shortest network path, up to 2 km, so the output is a connected route wherever the network allows it.

## Tuning without the test set

α (how much to trust the classifier) and β (how strictly the road distance must match the GPS distance) were tuned with Optuna (TPE, seed 42, 25 trials, 1.9 minutes). Tuning used the 80 training trips only and maximised reading-level Top-1. It used the stacking ensemble's **out-of-fold** probabilities, so every training trip was decoded with probabilities from models that never saw that trip. Student-test trips were then decoded once with the chosen values.

| Parameter | Search range | Default (trial 0) | Tuned |
| --- | --- | ---: | ---: |
| α | 0.1 – 5 (log) | 1.0 | 0.119 |
| β | 0.5 – 200 m (log) | 10 m | 7.57 m |

The objective is flat near the optimum: trials ranging from α 0.12 to 0.63 and β 3 to 12 m all reached 93.1 % training Top-1. The small α means that once the network constraint is applied, the classifier is best used as a gentle tie-breaker rather than a strong vote. A 7.6 m β means a candidate path that is 7.6 m longer or shorter than the GPS displacement costs one unit of log-probability.

On the training trips (out-of-fold), Top-1 rose from 85.4 % classifier-only to 92.0 % with the default α = 1, β = 10 m, and to 93.1 % with the tuned values.

## Classifier only versus classifier + HMM, student test

Both columns use the same route reconstruction, including gap filling. "Classifier only" takes the best-scoring arc at each reading.

| Probabilities from | Top-1 (classifier only) | Top-1 (+ HMM) | Route F1 | Route F1 (+ HMM) | Mismatch | Mismatch (+ HMM) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 85.21 % | 91.79 % | 0.830 | 0.986 | 0.448 | 0.028 |
| KNN (Optuna-tuned) | 85.67 % | 91.28 % | 0.833 | 0.987 | 0.436 | 0.028 |
| Decision Tree | 86.34 % | 92.03 % | 0.847 | 0.987 | 0.391 | 0.026 |
| Hist. Gradient Boosting (tuned) | 87.21 % | 93.58 % | 0.860 | 0.988 | 0.348 | 0.026 |
| Ensemble: soft vote | 86.46 % | 92.45 % | 0.847 | 0.987 | 0.390 | 0.028 |
| **Ensemble: stacking** | 87.18 % | **93.64 %** | 0.862 | 0.987 | 0.344 | 0.026 |

With stacking probabilities and the HMM, route precision is 0.978, recall 0.997 and continuity 0.999. Four Viterbi restarts occurred across the 10 trips, and on average 0.3 route gaps per trip could not be joined within 2 km. Decoding all 10 trips takes 0.47 s.

Per student-test trip, stacking + HMM (classifier-only in brackets):

| Trip | Top-1 | Route F1 | Mismatch |
| --- | ---: | ---: | ---: |
| 00000000 | 98.0 % (94.8) | 0.996 (0.955) | 0.008 (0.094) |
| 00000007 | 100.0 % (94.6) | 0.996 (0.977) | 0.007 (0.046) |
| 00000009 | 89.4 % (87.4) | 0.964 (0.872) | 0.074 (0.292) |
| 00000040 | 97.3 % (92.0) | 0.999 (0.904) | 0.002 (0.213) |
| 00000052 | 100.0 % (95.8) | 0.998 (0.942) | 0.003 (0.123) |
| 00000056 | 96.1 % (90.8) | 0.978 (0.904) | 0.046 (0.209) |
| 00000065 | 91.3 % (85.2) | 0.997 (0.762) | 0.006 (0.626) |
| 00000076 | 95.4 % (89.1) | 0.971 (0.715) | 0.060 (0.791) |
| 00000077 | 83.8 % (71.1) | 0.983 (0.768) | 0.034 (0.605) |
| 00000091 | 94.2 % (88.2) | 0.991 (0.820) | 0.018 (0.439) |

## Interpretation

- **The HMM is the largest single improvement in the project.** It adds 6 to 7 Top-1 points for every model and cuts route mismatch by more than 90 %. Tuning boosting added 0.2 points by comparison.
- **The classifier-only route is worse at route level than at reading level.** Every brief flip onto a wrong arc forces a detour when the route is joined into a connected path, which is why classifier-only route precision is only about 0.77. The HMM removes these flips, so precision rises to 0.98 while recall stays at 0.997.
- **After the HMM, the classifier matters for reading-level accuracy but hardly for the route.** The Top-1 gap between the best and worst model is 2.0 points before decoding and 2.4 points after it. Route F1, however, is 0.986 to 0.988 for every model. Stacking and tuned boosting remain the best probability sources.
- **Remaining errors are mostly at reading level, not route level.** Trip 77 still has the lowest Top-1 (83.8 %) but a route F1 of 0.983. The decoded route is right, but some readings are attributed to the neighbouring arc around junctions, where the derived labels are least certain.
- **Classifier-only and HMM numbers on the student test both use parameters fixed on training data.** No student-test number influenced α, β, the gap threshold or the search limit.

## Limits

- Reading-level Top-1 is measured against derived labels from the monotone route alignment, not against surveyed ground truth.
- The 60 s gap threshold, the transition search limit and the 2 km fill limit were set from the sampling statistics and were not tuned.
- Parallel arcs with identical endpoints are separate states, but they receive identical transition scores, so only the classifier can separate them.
