# Geospatial Mathematics

Coordinates are stored as longitude, latitude in WGS84. Haversine distance is the reference geodesic approximation. For local metric operations, each trip must choose a local projected CRS from its own bounds; one global UTM zone is invalid because the training trips span separated regions. Spatial indexes operate in projected metres. Bearing uses the initial great-circle bearing, and heading difference is the wrapped absolute difference in `[0, 180]` degrees.

The local projection and its EPSG code must be recorded with every derived artifact. Raw coordinates are preserved unchanged.
