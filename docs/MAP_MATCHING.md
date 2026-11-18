# Map Matching

Candidate generation first queries a metric spatial index, then computes point-to-segment distance and optional movement-heading difference. Results are ordered by `(distance, arc_id)` for deterministic ties. Search radius and candidate count are configuration values. Empty results are retained as explicit no-candidate observations. Large temporal gaps and zero movement suppress unreliable heading features rather than inventing a direction.
