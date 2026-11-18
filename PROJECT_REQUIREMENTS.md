# PathSnap Project Requirements

**Project:** PathSnap - Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A  
**Institution:** PES University  
**Status:** Source-of-truth requirements document; dataset-specific details remain subject to validation.

## 1. Project Overview

PathSnap is a machine-learning system that infers the most likely sequence of road-network arcs travelled by a vehicle from noisy GPS observations. Map-matching is the correspondence between each relevant GPS observation and plausible road segments. Route reconstruction is the sequential inference of a connected arc sequence that explains the observations over time.

The system must support a complete, reproducible workflow from raw trip files through validation, candidate generation, feature extraction, independently trained model comparison, probabilistic sequence decoding, evaluation, geographical visualisation, and evaluator-facing serving.

Successful completion means that the supplied data can be processed without fabricated inputs or results, at least five models are trained and evaluated under leakage-safe splits, a classifier-plus-HMM/Viterbi route can be produced, and the result can be selected and displayed through the two-laptop application architecture. Exact acceptance thresholds are **TBD** unless specified elsewhere in the authoritative documents.

The project is not a GPS tracker, a Google Maps replacement, a navigation engine, or a system that merely joins GPS points with straight lines.

## 2. Problem Understanding

GPS observations are noisy samples of a vehicle's position, not road paths. Errors may arise from measurement noise, signal interference, multipath effects near buildings, low sampling frequency, and GPS dropouts. A road network represents possible physical travel using nodes and directed or undirected connections (arcs). The system must identify nearby candidate road segments for observations and then use temporal and network consistency to reconstruct a plausible path.

Independent per-observation classification can produce impossible jumps between disconnected roads. Sequential reconstruction addresses this by modelling hidden road states, observation emissions, and transitions between compatible states. The HMM/Viterbi stage should select the highest-probability sequence rather than only the locally highest-probability candidate.

## 3. Dataset Specification

The expected trip-level inputs are:

| File | Meaning | Expected semantic contents |
|---|---|---|
| `.track` | GPS observations | Longitude, latitude, timestamp |
| `.nodes` | Road-network nodes | Node identifier and longitude/latitude |
| `.arcs` | Road-network connections | Arc identifier, `from_node_id`, `to_node_id` |
| `.route` | Reference route | Chronological `arc_id` sequence |

The repository must validate the actual headers, delimiters, encodings, indexing convention, and trip organisation before dataset-specific loaders are finalised. The currently assumed semantic contract is not permission to fabricate columns or values. Zero-based indexing must be preserved if present in the supplied data; if the files use another convention, the convention must be recorded rather than silently changed.

The required partitions are:

- Training: 80 trips.
- Student testing: 10 trips for development-stage comparison.
- Evaluator testing: 10 trips reserved for evaluator use.

Partition membership must be fixed and versioned. GPS points from one trip must not be split across training and validation folds. Evaluator-testing observations and their reference routes must not influence feature design, model selection, hyperparameter optimisation, thresholds, or reported development decisions.

Validation must include file existence, header/schema checks, missing values, numeric parsing, timestamp validity and ordering, coordinate bounds, duplicate or malformed identifiers, valid arc endpoints, network references, and route references. The required bounds are longitude in `[-180, 180]` and latitude in `[-90, 90]`. Road-network connectivity, disconnected components, and isolated nodes must be measured from the actual data.

Reference `.route` files are labels for evaluation and analysis. They are not prediction inputs and must remain backend/evaluation-side assets.

## 4. Data Exploration Requirements

The exploration phase must report, using actual data:

- trip inventory, file availability, and file sizes;
- GPS observation counts, data types, missing values, invalid coordinates, timestamp ordering, and irregular time gaps;
- geographic extent and trajectory distributions;
- road-network node/arc counts, connectivity, disconnected components, and isolated nodes;
- reference-route availability and basic route structure.

Folium maps and trajectory visualisations are required. Exploration outputs must identify their source data and must not imply metrics for data that has not been loaded.

## 5. System Architecture

```text
Raw Data
  -> Data Validation
  -> Preprocessing
  -> Road Candidate Generation
  -> Feature Engineering
  -> ML Model Training
  -> Probability Estimation
  -> HMM
  -> Viterbi Decoding
  -> Route Reconstruction
  -> Evaluation
  -> FastAPI
  -> Frontend
  -> Geographical Visualisation
```

Responsibilities:

- **Validation/preprocessing:** establish trustworthy types, ordering, identifiers, coordinates, and network references.
- **Candidate generation:** find plausible road segments for each observation and measure candidate miss rate.
- **Feature engineering:** represent spatial, directional, temporal, and network evidence without leakage.
- **Model training:** fit and compare independently configured models using only the permitted training partition and group-aware validation.
- **Probability estimation:** expose calibrated or model-derived candidate scores for sequential inference where supported.
- **HMM/Viterbi:** enforce probabilistic temporal and connectivity consistency.
- **Evaluation:** compare predictions with backend-only reference routes and report permitted metrics.
- **FastAPI:** load the selected trip, execute the pipeline, and return prediction results.
- **Frontend:** allow evaluator selection and display returned routes, maps, and permitted metrics.
- **Visualisation:** render interpretable interactive geographical layers.

Implementation details not stated as mandatory in the source documents are recommendations or **TBD**, not requirements.

## 6. Feature Engineering Requirements

The feature component may include the following proposed features; these are implementation candidates, not individually mandated source requirements:

- **Spatial:** GPS-to-segment projection, nearest-point geometry, local coordinates.
- **Distance:** point-to-segment distance, along-segment distance, network distance.
- **Directional:** GPS bearing, segment bearing, bearing difference, direction consistency.
- **Temporal:** time delta, implied speed, sampling-gap indicators.
- **Road-network:** node degrees, arc connectivity, component identifiers, segment length.
- **Sequential:** previous/next candidate identity, transition distance, turn or continuity indicators.

Candidate-to-road correspondence must be explicit. Candidate miss rate must be measured against the available reference route or another clearly documented ground-truth correspondence. Any feature that uses labels, future observations, or evaluator routes must be identified and excluded from prediction-time inputs.

## 7. Machine Learning Requirements

At least five independently trained and evaluated models are mandatory. The 80 training trips must be used for fitting and group-aware cross-validation. The 10 student-testing trips are for development-stage comparison. All configurations, seeds, data versions, preprocessing choices, training/inference times where practical, logs, and results must be retained.

Potential algorithm families include Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, SVM, and MLP. This list is exploratory; every listed algorithm is not mandatory. The final model must be selected from actual measured results and justified.

No dataset, labels, metrics, or model superiority may be invented.

## 8. Evaluation Metrics

The consolidated comparison should cover, where applicable:

- accuracy, precision, recall, macro F1, confusion matrix, and log loss;
- observation-level Top-1 Accuracy;
- candidate miss rate;
- route continuity and sequence-level performance;
- training time and inference time.

Accuracy measures correct predictions; precision measures correctness among predicted classes; recall measures recovered true classes; macro F1 averages class-wise F1 equally; a confusion matrix exposes class-specific errors; log loss evaluates probabilistic confidence; Top-1 accuracy measures the highest-ranked candidate; candidate miss rate measures cases where the reference is absent from candidates; route continuity measures network-consistent sequencing; sequence-level performance evaluates the reconstructed route as a sequence. Exact route-distance and sequence metrics are **TBD** where not defined by the source documents.

A consolidated model-comparison table is mandatory. It must contain actual results and clearly identify the split and evaluation protocol.

## 9. Bayesian Hyperparameter Optimisation

Bayesian hyperparameter optimisation using Optuna is mandatory. The study must document the search space, ranges, objective, group-aware cross-validation protocol, trial logs, selected parameters, computational cost, and final refitting procedure. Tuned and untuned results must be compared using the same leakage-safe evaluation design.

The evaluator-testing partition must never influence search spaces, trial outcomes, parameter selection, stopping decisions, or final model selection. Exact parameter ranges, trial counts, sampler, pruner, and objective direction are **TBD** until data scale and computational constraints are known.

## 10. HMM and Viterbi Requirements

The implementation must document:

- hidden states as candidate road segments or another explicitly justified road-state representation;
- emission probabilities linking GPS observations to candidate segments;
- transition probabilities linking consecutive compatible road states;
- the Viterbi recurrence and back-pointer reconstruction;
- candidate-to-road correspondence and missing-candidate handling;
- invalid transitions, disconnected segments, and large GPS gaps;
- route continuity and failure handling.

The decoder must produce a predicted road segment for each relevant GPS observation. Classifier-only and classifier-plus-HMM/Viterbi behaviour should be compared. Exact transition model, gap policy, and smoothing strategy are **TBD** until the real network and sampling characteristics are inspected.

## 11. Geographical Visualisation

Mandatory Folium outputs must be able to show:

- original GPS observations;
- relevant road-network geometry;
- reference route where available;
- predicted route;
- distinguishable colours, markers, labels or legends;
- appropriate zoom levels and layer controls where appropriate.

Maps must be interactive HTML files. Representative screenshots must be included in the technical report. Ground-truth layers must not be exposed in the evaluator prediction response unless the permitted evaluation design explicitly allows a derived metric or view.

## 12. Backend Requirements

The FastAPI backend must accept a trip ID, load the selected trip, execute candidate generation and feature extraction, run model inference, perform HMM/Viterbi decoding, and return the predicted route plus permitted evaluation metrics. It must load versioned model artifacts and configuration, validate requests, fail clearly when artifacts are absent, and keep reference labels server-side.

Endpoint names, authentication, production hosting, CORS policy, and serialization format are **TBD** unless specified by the deployment environment.

## 13. Frontend Requirements

The evaluator-facing React UI must display ten reserved evaluation-trip options, send the selected trip ID to the API, and display the reconstructed route, geographical map, and returned evaluation results. It must not receive, bundle, infer, or expose ground-truth labels. The exact component library, styling, route format, and loading/error states are **TBD**.

## 14. Two-Laptop Deployment

**Laptop 1** must host the trained classifier, HMM/Viterbi implementation, road-network information, FastAPI backend, and server-side evaluation assets.

**Laptop 2** must host the evaluator-facing UI, display ten evaluation-trip options, send the selected trip ID to Laptop 1, and display the returned results.

Communication must occur over a documented network API. The concrete IP/hostname, port, firewall configuration, CORS configuration, startup commands, and whether the laptops share a local network or tunnel are **TBD**. The evaluator must not need direct access to backend ground-truth files.

## 15. Deliverables and Assessment Controls

The expected submission includes a public repository containing source code, trained models, Bayesian optimisation experiments, HMM/Viterbi implementation, FastAPI backend, React frontend, Folium visualisations, and setup documentation, plus a consolidated technical PDF covering exploration, feature engineering, model experiments, comparison, optimisation, HMM, route reconstruction, maps, and deployment evidence.

Assessment and plagiarism requirements from the policy documents are binding. All reused code, data, figures, and external references must be attributed according to the course policy. Work must be original, reproducible, and auditable. Exact marking weights, submission dates, formatting rules, and permitted collaboration boundaries are **TBD** here unless transcribed directly from the policy document.

## 16. Development Guardrails

- Do not train until the actual dataset schema and partitions are inspected.
- Do not fabricate samples, labels, metrics, screenshots, or experiment results.
- Split by trip/group, never randomly by GPS observation.
- Keep evaluator ground truth backend-only.
- Preserve configurations, seeds, versions, logs, and results.
- Record assumptions and unresolved ambiguities.
- Add tests for validation, feature computation, candidate generation, decoding, API boundaries, and leakage controls as each component is implemented.

## 17. Audit Coverage Report

This audit treats each explicit requirement family in the five source documents as one auditable mandatory requirement when it has a distinct deliverable, constraint, or acceptance condition. Sub-requirements are listed below so future changes can be checked without silently broadening the specification.

| ID | Mandatory requirement audited | Status |
|---|---|---|
| M01 | Define the map-matching and route-reconstruction problem | Documented |
| M02 | Explain GPS noise, interference, multipath, low sampling, and dropouts | Documented |
| M03 | Describe raw GPS observations versus road-network paths | Documented |
| M04 | Document `.track` inputs | Documented |
| M05 | Document `.nodes` inputs | Documented |
| M06 | Document `.arcs` inputs | Documented |
| M07 | Document `.route` inputs | Documented |
| M08 | Preserve the supplied indexing convention, including zero-based indexing where applicable | Documented |
| M09 | Use the 80-trip training partition | Documented |
| M10 | Use the 10-trip student-testing partition for development comparison | Documented |
| M11 | Reserve the 10-trip evaluator-testing partition | Documented |
| M12 | Validate files, schemas, missing values, timestamps, coordinates, identifiers, references, and connectivity | Documented |
| M13 | Prevent trip-level data leakage | Documented |
| M14 | Keep evaluator ground truth backend-only | Documented |
| M15 | Perform the required dataset exploration analyses | Documented |
| M16 | Produce Folium maps and trajectory visualisations | Documented |
| M17 | Implement the complete processing architecture | Documented |
| M18 | Separate mandatory features from proposed implementation features | Documented |
| M19 | Train and evaluate at least five independent models | Documented |
| M20 | Fit/cross-validate using the training partition and compare on student testing | Documented |
| M21 | Record configurations, timings where practical, logs, and results | Documented |
| M22 | Compare models using actual results and justify selection | Documented |
| M23 | Use Bayesian hyperparameter optimisation with Optuna | Documented |
| M24 | Log search spaces, trials, objective, group-aware CV, selection, refit, and cost | Documented |
| M25 | Exclude evaluator testing from optimisation and selection | Documented |
| M26 | Report classification metrics and confusion matrix | Documented |
| M27 | Report Top-1 accuracy and candidate miss rate | Documented |
| M28 | Report route continuity and sequence-level performance | Documented |
| M29 | Report training and inference time where practical | Documented |
| M30 | Provide a consolidated model-comparison table | Documented |
| M31 | Implement HMM hidden states, emissions, transitions, and Viterbi decoding | Documented |
| M32 | Handle missing candidates, invalid transitions, gaps, disconnected roads, and continuity | Documented |
| M33 | Produce a road-segment prediction for each relevant observation | Documented |
| M34 | Compare classifier-only with classifier-plus-HMM/Viterbi output | Documented |
| M35 | Produce required Folium layers, HTML maps, and report screenshots | Documented |
| M36 | Provide the FastAPI prediction/evaluation workflow | Documented |
| M37 | Provide the evaluator-facing React workflow without exposing labels | Documented |
| M38 | Deploy with the specified Laptop 1/Laptop 2 responsibilities | Documented |
| M39 | Submit the required repository contents and consolidated technical PDF | Documented |
| M40 | Follow assessment, attribution, originality, and plagiarism controls | Documented as policy-controlled; exact policy values remain TBD |

**Total mandatory requirements identified:** 40  
**Requirements documented:** 40  
**Requirements missing:** 0 identified from the available source requirements.  
**Ambiguous requirements requiring clarification:** 8 areas: exact raw-file delimiters and headers; exact route/observation alignment; exact metric formulas for route and sequence performance; candidate search radius/count; exact Optuna search spaces and trial budget; HMM transition/emission parameterisation and large-gap policy; API contract/network configuration; assessment-policy weights, dates, citation format, and permitted collaboration boundaries.

The ambiguous items are intentionally marked **TBD** elsewhere in this document. They must be resolved from the supplied dataset, the exact policy text, or an explicit instructor clarification before implementation choices depend on them.
