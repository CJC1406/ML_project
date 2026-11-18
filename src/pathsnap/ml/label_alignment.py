from dataclasses import dataclass
import math
from shapely.geometry import Point
from shapely.strtree import STRtree


@dataclass(frozen=True)
class Alignment:
    observation_index: int
    route_position: int | None
    arc_id: int | None
    distance_m: float | None
    confidence: float
    ambiguous: bool
    reason: str
    sequence_cost: float | None = None
    alternative_cost: float | None = None


def align_observations(track, route_arc_ids, arc_geometries, projection, max_distance_m=100.0, ambiguity_margin_m=5.0, gap_penalty=25.0, max_states=12, gap_seconds=60.0):
    """Bounded ordered DP over route occurrences; spatial query limits states to K."""
    if not route_arc_ids or not len(track): return []
    if len(route_arc_ids) != len(arc_geometries): raise ValueError("route occurrence IDs and geometries must have equal length")
    tree = STRtree(arc_geometries); state_rows=[]
    for _, row in track.reset_index(drop=True).iterrows():
        xy=projection.forward.transform(float(row.longitude),float(row.latitude)); point=Point(xy)
        hits=tree.query(point.buffer(max_distance_m))
        ranked=sorted(((float(point.distance(arc_geometries[int(h)])),int(h)) for h in hits), key=lambda x:(x[0],x[1]))[:max_states]
        state_rows.append(ranked)
    costs=[]; backs=[]; emissions=[]
    for i, states in enumerate(state_rows):
        emissions.append({j:d for d,j in states}); current={}; pointers={}
        if i==0:
            current={j:d for d,j in states}; pointers={j:None for _,j in states}
        else:
            for j,d in states:
                options=[(pc+d+gap_penalty*max(0,j-k-1),k) for k,pc in costs[-1].items() if k<=j]
                # Keep an explicit recoverable state when a GPS gap or route
                # discontinuity breaks all monotone predecessors.
                options.append((d + max_distance_m * 10, None))
                if options:
                    best_option=min(options,key=lambda item:item[0]); current[j],pointers[j]=best_option
        costs.append(current); backs.append(pointers)
    if not costs[-1]: return [Alignment(i,None,None,None,0.0,False,"no_valid_path") for i in range(len(track))]
    final=min(costs[-1],key=costs[-1].get); positions=[final]
    for i in range(len(track)-1,0,-1):
        previous=backs[i].get(positions[-1]); positions.append(positions[-1] if previous is None else previous)
    positions.reverse(); result=[]
    for i,j in enumerate(positions):
        if j is None or j not in emissions[i]: result.append(Alignment(i,None,None,None,0.0,False,"no_valid_path")); continue
        local=emissions[i][j]; seq=costs[i][j]; alternatives=sorted(d for p,d in emissions[i].items() if p!=j)
        alt=alternatives[0] if alternatives else float('inf'); ambiguous=bool(alternatives and abs(alt-local)<=ambiguity_margin_m)
        confidence=max(0.0,min(1.0,math.exp(-local/max_distance_m))) if local<=max_distance_m else 0.0
        result.append(Alignment(i,j,int(route_arc_ids[j]),local,confidence,ambiguous,"ambiguous" if ambiguous else ("low_confidence" if confidence==0 else "aligned"),seq,alt))
    return result
