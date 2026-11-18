import numpy as np
from pathsnap.geo.distance import bearing_deg, heading_difference_deg, haversine_m


FEATURE_NAMES = ["distance_m", "heading_difference_deg", "speed_mps", "time_gap_s", "movement_m", "segment_length_m", "distance_to_start_m", "distance_to_end_m", "start_degree", "end_degree"]


def candidate_features(track, road, candidates, graph=None):
    rows=[]
    for c in candidates:
        obs=track.iloc[c.observation_index]; geom=road.geometries[road.arcs.index[road.arcs.arc_id == c.arc_id][0]]
        start,end=geom.coords[0],geom.coords[-1]; prev=track.iloc[c.observation_index-1] if c.observation_index else None
        gap=float(obs.timestamp-prev.timestamp) if prev is not None else np.nan
        movement=haversine_m(float(prev.longitude),float(prev.latitude),float(obs.longitude),float(obs.latitude)) if prev is not None else np.nan
        speed=movement/gap if prev is not None and gap>0 else np.nan
        rows.append({"observation_index":c.observation_index,"arc_id":c.arc_id,"distance_m":c.distance_m,"heading_difference_deg":c.heading_difference_deg if c.heading_difference_deg is not None else np.nan,"speed_mps":speed,"time_gap_s":gap,"movement_m":movement,"segment_length_m":geom.length,"distance_to_start_m":0.0,"distance_to_end_m":0.0,"start_degree":np.nan,"end_degree":np.nan})
    return rows
