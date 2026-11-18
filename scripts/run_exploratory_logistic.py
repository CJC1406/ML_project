from pathlib import Path
import json, os, time
import pandas as pd
from pathsnap.data.loader import discover_trip_directories, load_trip
from pathsnap.geo.geometry import build_road_geometry
from pathsnap.map_matching.candidate_generation import generate_candidates
from pathsnap.ml.label_alignment import align_observations
from pathsnap.ml.training_dataset import make_labelled_candidates
from pathsnap.training.trainer import train_exploratory


def build_partition(root: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    frames=[]; summary=[]
    for directory in discover_trip_directories(root):
        trip=load_trip(directory); road=build_road_geometry(trip.nodes,trip.arcs)
        route_ids=trip.route.arc_id.astype(int).tolist()
        route_geoms=[]
        for arc_id in route_ids:
            matches=road.arcs.index[road.arcs.arc_id==arc_id].tolist()
            if matches: route_geoms.append(road.geometries[matches[0]])
        align=align_observations(trip.track,route_ids,route_geoms,road.projection)
        candidates=generate_candidates(trip.track,road,radius_m=50,max_candidates=8)
        labelled=make_labelled_candidates(trip.track,road,candidates,align)
        if not labelled.frame.empty:
            labelled.frame["trip_id"]=trip.trip_id; frames.append(labelled.frame)
        summary.append({"trip_id":trip.trip_id,"track_rows":len(trip.track),"candidates":len(candidates),"usable":labelled.usable_observations,"ambiguous":labelled.ambiguous_observations,"unaligned":labelled.unaligned_observations,"candidate_misses":labelled.candidate_misses})
        print(trip.trip_id, len(candidates), flush=True)
    frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(summary).to_csv(out/"alignment_summary.csv",index=False); return frame,{"trips":summary}


if __name__ == "__main__":
    train_root=Path(os.environ["PATHSNAP_TRAIN_ROOT"]); student_root=Path(os.environ["PATHSNAP_STUDENT_TEST_ROOT"])
    base=Path("experiments/local/exploratory_logistic"); base.mkdir(parents=True,exist_ok=True)
    config={"experiment":"EXPLORATORY","radius_m":50,"max_candidates":8,"alignment_max_distance_m":100,"random_state":42,"train_root":str(train_root),"student_test_root":str(student_root)}
    (base/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    train_frame,train_summary=build_partition(train_root,base/"train")
    if train_frame.empty: raise SystemExit("No usable training candidates")
    result=train_exploratory(train_frame,base/"model")
    (base/"summary.json").write_text(json.dumps({"training":train_summary,"metrics":result},indent=2,default=str),encoding="utf-8")
    print(json.dumps(result,indent=2,default=str))
