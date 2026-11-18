from dataclasses import dataclass
import pandas as pd
from .features import candidate_features, FEATURE_NAMES


@dataclass
class LabelledCandidates:
    frame: pd.DataFrame
    usable_observations: int
    ambiguous_observations: int
    unaligned_observations: int
    candidate_misses: int


def make_labelled_candidates(track, road, candidates, alignments):
    aligned={a.observation_index:a for a in alignments}
    rows=candidate_features(track,road,candidates)
    for row in rows:
        a=aligned.get(row["observation_index"])
        row["alignment_status"] = a.reason if a else "unaligned"
        row["candidate_miss"] = bool(a and a.arc_id is not None and a.arc_id not in {x["arc_id"] for x in rows if x["observation_index"] == a.observation_index})
        row["label"] = int(a is not None and not a.ambiguous and a.arc_id == row["arc_id"] and a.confidence > 0)
    frame=pd.DataFrame(rows)
    usable=sum(a.confidence>0 and not a.ambiguous for a in alignments); ambiguous=sum(a.ambiguous for a in alignments); unaligned=sum(a.route_position is None for a in alignments); misses=sum(a.arc_id is not None and a.arc_id not in {x["arc_id"] for x in rows if x["observation_index"] == a.observation_index} for a in alignments)
    if not frame.empty: frame=frame[frame.alignment_status.isin(["aligned","low_confidence"])].copy()
    return LabelledCandidates(frame,usable,ambiguous,unaligned,misses)
