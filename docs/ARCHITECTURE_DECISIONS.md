# Architecture Decisions

- `src/pathsnap` is the canonical package boundary.
- Data, geometry, graph, matching, decoding, API, and visualisation remain separate modules.
- Raw and evaluator data never enter Git.
- Numeric track timestamps remain elapsed seconds.
- Per-trip projection is mandatory for metric spatial work.
- Route alignment is a research step and is not assumed to be one label per GPS point.
- Viterbi uses observation-specific states and exact zero-probability handling.
