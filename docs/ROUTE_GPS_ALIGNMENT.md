# Route/GPS Alignment Design

The training files contain a GPS sequence and a separate chronological arc sequence; their lengths differ, so route entries are not labels for individual GPS rows.

Candidate approaches:

1. **Projection:** project each GPS point to reference-route geometry. Simple, but can skip arcs at intersections or during gaps.
2. **Ordered sequence alignment:** align GPS observations to the ordered route with emissions from spatial distance and transitions from arc adjacency. Handles repeated observations and multiple arcs between observations, but requires gap penalties.
3. **Network-constrained dynamic programming:** allow zero or more route arcs between consecutive observations and score the shortest connected network explanation. More faithful for large gaps, but more expensive.

Recommendation: use ordered network-constrained dynamic programming as the research label-alignment method, validate it on multiple training trips with manual/map inspection, and only then generate supervised labels. Until verified, route files remain evaluation references rather than observation labels.
