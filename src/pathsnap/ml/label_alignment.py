from dataclasses import dataclass
import math
from shapely.geometry import Point


@dataclass(frozen=True)
class Alignment:
    observation_index: int
    route_position: int | None
    arc_id: int | None
    distance_m: float | None
    confidence: float
    ambiguous: bool
    reason: str


def align_observations(track, route_arc_ids, arc_geometries, projection, max_distance_m: float = 100.0, ambiguity_margin_m: float = 5.0, gap_penalty: float = 25.0) -> list[Alignment]:
    """Ordered dynamic program over route *occurrences*, not unique arc IDs.

    States are (observation index, route position). Staying on an occurrence costs
    geometric distance; advancing costs distance plus a configurable skip penalty.
    Route discontinuities are represented by the resulting high cost and are never
    silently repaired.
    """
    if not route_arc_ids or not len(track): return []
    n, m = len(track), len(route_arc_ids)
    costs = [[float("inf")] * m for _ in range(n)]
    back = [[None] * m for _ in range(n)]
    for i, row in track.reset_index(drop=True).iterrows():
        point = projection.forward.transform(float(row.longitude), float(row.latitude))
        for j, geom in enumerate(arc_geometries):
            d = geom.distance(Point(point))
            if i == 0: costs[i][j] = d
            else:
                candidates = [(costs[i-1][k] + d + gap_penalty * max(0, j-k-1), k) for k in range(j+1) if math.isfinite(costs[i-1][k])]
                if candidates: costs[i][j], back[i][j] = min(candidates)
    end = min(range(m), key=lambda j: costs[-1][j])
    if not math.isfinite(costs[-1][end]): return [Alignment(i, None, None, None, 0.0, False, "no_valid_path") for i in range(n)]
    positions = [end]
    for i in range(n-1, 0, -1): positions.append(back[i][positions[-1]])
    positions.reverse()
    result=[]
    for i,j in enumerate(positions):
        if j is None: result.append(Alignment(i,None,None,None,0.0,False,"no_valid_path")); continue
        d=costs[i][j]
        result.append(Alignment(i,j,int(route_arc_ids[j]),d,math.exp(-d/max_distance_m),d>max_distance_m,"distance" if d<=max_distance_m else "low_confidence"))
    return result
