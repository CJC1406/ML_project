from dataclasses import dataclass
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from .projection import choose_trip_projection, project_xy


@dataclass
class RoadGeometry:
    arcs: pd.DataFrame
    geometries: list[LineString]
    index: STRtree
    projection: object | None = None
    geographic_geometries: list[LineString] | None = None


def build_road_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame) -> RoadGeometry:
    node_lookup = {int(r.node_id if "node_id" in nodes else i): (float(r.longitude), float(r.latitude)) for i, r in nodes.reset_index(drop=True).iterrows()}
    projection = choose_trip_projection(nodes.longitude.tolist(), nodes.latitude.tolist())
    geometries = []
    geographic_geometries = []
    kept = []
    for _, row in arcs.iterrows():
        start = node_lookup.get(int(row.from_node_id)); end = node_lookup.get(int(row.to_node_id))
        if start is None or end is None:
            continue
        kept.append(row)
        geographic_geometries.append(LineString([start, end]))
        geometries.append(LineString([project_xy(projection, *start), project_xy(projection, *end)]))
    frame = pd.DataFrame(kept).reset_index(drop=True)
    return RoadGeometry(frame, geometries, STRtree(geometries) if geometries else STRtree([]), projection, geographic_geometries)


def query_geometry(road: RoadGeometry, point: Point, radius_m: float) -> list[int]:
    if not road.geometries:
        return []
    hits = road.index.query(point.buffer(radius_m))
    return [int(i) for i in hits]
