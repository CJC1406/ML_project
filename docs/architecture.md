# Architecture

Laptop 1 is the trusted backend. It owns raw tracks, road-network data, route labels, trained models, experiment artifacts, and evaluation. Laptop 2 serves the React evaluator. It may request a reserved trip ID and receive predictions, metrics, and a rendered map, but never receives route labels.

The API boundary will expose prediction objects rather than files containing ground truth. Training and evaluation splits are by trip/group, never by individual GPS point.
