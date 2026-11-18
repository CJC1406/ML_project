from dataclasses import dataclass
from shapely.geometry import Point
from pathsnap.geo.distance import bearing_deg, heading_difference_deg, haversine_m
from pathsnap.geo.geometry import RoadGeometry, query_geometry


@dataclass(frozen=True)
class Candidate:
    observation_index: int
    arc_id: int
    distance_m: float
    heading_difference_deg: float | None


def generate_candidates(track, road: RoadGeometry, radius_m: float = 50.0, max_candidates: int | None = 8) -> list[Candidate]:
    output = []
    for obs_idx, row in track.reset_index(drop=True).iterrows():
        point = Point(*road.projection.forward.transform(float(row.longitude), float(row.latitude)))
        heading = None
        if obs_idx > 0:
            prev = track.iloc[obs_idx - 1]
            dt = float(row.timestamp) - float(prev.timestamp)
            movement = haversine_m(float(prev.longitude), float(prev.latitude), float(row.longitude), float(row.latitude))
            if 0 < dt <= 60 and movement >= 1.0:
                heading = bearing_deg(float(prev.longitude), float(prev.latitude), float(row.longitude), float(row.latitude))
        ranked = []
        for geometry_idx in query_geometry(road, point, radius_m):
            geom = road.geometries[geometry_idx]
            distance = point.distance(geom)
            if distance <= radius_m:
                geographic = road.geographic_geometries[geometry_idx]
                segment_heading = bearing_deg(*geographic.coords[0], *geographic.coords[-1])
                diff = heading_difference_deg(heading, segment_heading) if heading is not None else None
                ranked.append((distance, int(road.arcs.iloc[geometry_idx].arc_id), diff))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if max_candidates is not None:
            ranked = ranked[:max_candidates]
        output.extend(Candidate(obs_idx, arc_id, distance, diff) for distance, arc_id, diff in ranked)
    return output
