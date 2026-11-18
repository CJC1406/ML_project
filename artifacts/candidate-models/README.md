# Fitted PathSnap candidate models

The two Joblib artifacts were refitted on all 80 training trips with the original comparison settings. Reloading each artifact produced identical probabilities on the scored student-test candidates. Student-test Top-1 was 85.32% for Decision Tree and 85.44% for Histogram Gradient Boosting. See the [experiment report](../../docs/CANDIDATE_MODEL_COMPARISON.md) for the protocol and limits of the provisional route labels.

Use the scikit-learn version and ordered feature list in [manifest.json](manifest.json). Each estimator accepts a numeric matrix with one row per candidate and returns probabilities through `predict_proba`. The positive class is `1`. The Decision Tree artifact includes its fitted median imputer. Histogram Gradient Boosting handles missing feature values directly.

The [training script](../../scripts/compare_candidate_models.py) also contains the feature calculations. The current FastAPI registry uses the Logistic Regression feature pipeline. To serve these two models, register the artifacts and feed them the six features in the manifest order using the feature calculations from the comparison.

To reproduce the exported artifacts from the repository root:

```bash
python scripts/compare_candidate_models.py "/path/to/Student folder" --out data/processed/candidate_comparison.json --artifacts-dir artifacts/candidate-models
```
