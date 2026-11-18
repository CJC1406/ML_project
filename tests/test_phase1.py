import math
import pandas as pd
from src.data.loader import TripData
from src.data.validator import validate_trip
from src.geo.distance import bearing_deg, haversine_m, heading_difference_deg
from src.geo.road_network import build_graph, graph_statistics


def trip():
    return TripData("x", pd.DataFrame([[0, 0, "2024-01-01T00:00:00Z"], [0.001, 0, "2024-01-01T00:00:01Z"]], columns=["longitude", "latitude", "timestamp"]), pd.DataFrame([[0,0],[.001,0]], columns=["longitude","latitude"]), pd.DataFrame([[0,1]], columns=["from_node_id","to_node_id"]), pd.DataFrame([[0]], columns=["arc_id"]))


def test_validation_and_graph():
    assert validate_trip(trip()).valid
    g = build_graph(trip().nodes, trip().arcs)
    assert graph_statistics(g)["arcs"] == 1


def test_distance_and_bearing():
    assert math.isclose(haversine_m(0, 0, 0, 1), 111195, rel_tol=.002)
    assert math.isclose(bearing_deg(0, 0, 1, 0), 90, abs_tol=.01)
    assert heading_difference_deg(350, 10) == 20
