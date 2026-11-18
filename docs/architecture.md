# Architecture

Laptop 1 is the trusted backend. It owns raw tracks, road-network data, route labels, trained models, experiment artifacts, and evaluation. Laptop 2 serves the React evaluator. It may request a reserved trip ID and receive predictions, metrics, and a rendered map, but never receives route labels.

The API boundary will expose prediction objects rather than files containing ground truth. Training and evaluation splits are by trip/group, never by individual GPS point.

## Phase 2 geospatial path

The loader accepts headered or whitespace/comma/semicolon/tab-separated files and
normalises them into typed semantic columns. Numeric track timestamps are elapsed
seconds, not Unix epoch values; text timestamps are parsed only when present. Validation rejects malformed coordinates,
timestamps, references, and non-monotonic tracks.

Road geometry maps each preserved arc ID to a directed `LineString`. A `MultiDiGraph`
is used because distinct parallel directed arcs are meaningful. Candidate generation
uses Shapely `STRtree` for coarse spatial filtering, then computes point-to-segment
distance and optional heading difference, returning deterministic `(observation_index,
arc_id)` candidates. Query-envelope distance is an approximation in degrees until the
dataset region is known; metric CRS selection is deliberately deferred.
