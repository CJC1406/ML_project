from dataclasses import dataclass
from shapely.geometry import Point
from pathsnap.geo.distance import bearing_deg, heading_difference_deg, point_to_segment_m
from pathsnap.geo.geometry import RoadGeometry, query_geometry


@dataclass(frozen=True)
class Candidate:
    observation_index: int
    arc_id: int
    distance_m: float
    heading_difference_deg: float | None


def generate_candidates(track, road: RoadGeometry, radius_m: float = 50.0, max_candidates: int | None = 8) -> list[Candidate]:
    """Generate deterministic candidates; radius uses a conservative degree conversion.

    The conversion is only a query envelope. Distances are calculated in metres with
    the local equirectangular approximation in ``distance.py``. A projected CRS should
    replace this once the dataset region is known.
    """
    output = []
    radius_deg = radius_m / 111_320.0
    for obs_idx, row in track.reset_index(drop=True).iterrows():
        point = Point(float(row.longitude), float(row.latitude))
        heading = getattr(row, "heading_deg", None)
        ranked = []
        for geometry_idx in query_geometry(road, point, radius_deg):
            geom = road.geometries[geometry_idx]
            start, end = geom.coords[0], geom.coords[-1]
            distance = point_to_segment_m((point.x, point.y), start, end)
            if distance <= radius_m:
                segment_heading = bearing_deg(*start, *end)
                diff = heading_difference_deg(float(heading), segment_heading) if heading is not None else None
                ranked.append((distance, int(road.arcs.iloc[geometry_idx].arc_id), diff))
        ranked.sort(key=lambda x: (x[0], x[1]))
        if max_candidates is not None: ranked = ranked[:max_candidates]
        output.extend(Candidate(obs_idx, arc_id, dist, diff) for dist, arc_id, diff in ranked)
    return output
