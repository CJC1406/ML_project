# Viva Preparation

PathSnap does not classify GPS points in isolation. It creates road candidates, extracts spatial and temporal evidence, and uses graph-valid sequential decoding to reconstruct a route. The training data has headerless whitespace files: row order supplies node and arc identity, while track timestamps are elapsed seconds.

The main risks are timestamp misinterpretation, collapsed parallel arcs, invalid metric calculations across regions, route/GPS misalignment, and leakage from reference routes. Each is addressed by validation, per-trip projection, `MultiDiGraph`, explicit alignment research, and backend-only labels.
