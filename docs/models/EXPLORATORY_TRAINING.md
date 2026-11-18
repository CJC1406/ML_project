# Exploratory Logistic Regression

This phase is explicitly exploratory. It uses the real training and student-test partitions, but route-to-GPS alignment has not been independently verified. Any metrics are agreement with provisional aligned labels, not final GPS-to-road accuracy.

The pipeline uses ordered route-occurrence alignment, candidate features, a `StandardScaler` plus `LogisticRegression` pipeline, and trip-level `GroupKFold`. Student-test data is not used for fitting, scaling, feature selection, or tuning.

The route position and arc ID are labels/alignment metadata only; they must never be model features. Ambiguous, unaligned, and candidate-miss observations are recorded separately.
