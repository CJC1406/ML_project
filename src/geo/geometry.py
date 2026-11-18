"""Compatibility import; implementation lives in pathsnap.geo.geometry."""
from pathsnap.geo.geometry import *


@dataclass
class RoadGeometry:
    arcs: pd.DataFrame
    geometries: list[LineString]
    index: STRtree


def build_road_geometry(nodes: pd.DataFrame, arcs: pd.DataFrame) -> RoadGeometry:
    node_lookup = {int(r.node_id if "node_id" in nodes else i): (float(r.longitude), float(r.latitude)) for i, r in nodes.reset_index(drop=True).iterrows()}
    geometries = []
    kept = []
    for _, row in arcs.iterrows():
        start = node_lookup.get(int(row.from_node_id)); end = node_lookup.get(int(row.to_node_id))
        if start is None or end is None:
            continue
        kept.append(row)
        geometries.append(LineString([start, end]))
    frame = pd.DataFrame(kept).reset_index(drop=True)
    return RoadGeometry(frame, geometries, STRtree(geometries) if geometries else STRtree([]))


def query_geometry(road: RoadGeometry, point: Point, radius_degrees: float) -> list[int]:
    if not road.geometries:
        return []
    hits = road.index.query(point.buffer(radius_degrees))
    return [int(i) for i in hits]
