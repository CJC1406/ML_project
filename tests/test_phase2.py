import pandas as pd
from pathsnap.data.validator import parse_timestamps
from pathsnap.geo.road_network import build_graph
from pathsnap.geo.geometry import build_road_geometry
from src.map_matching.candidate_generation import generate_candidates


def frames():
    nodes = pd.DataFrame([[0, 0, 0], [1, .001, 0], [2, .001, .001]], columns=["node_id", "longitude", "latitude"])
    arcs = pd.DataFrame([[10, 0, 1], [11, 0, 1], [12, 1, 2]], columns=["arc_id", "from_node_id", "to_node_id"])
    track = pd.DataFrame([[.0005, 0, 1700000000]], columns=["longitude", "latitude", "timestamp"])
    return nodes, arcs, track


def test_numeric_timestamps_are_seconds():
    assert parse_timestamps(pd.Series([1700000000.25])).iloc[0] == 1700000000.25


def test_elapsed_timestamp_edge_cases():
    assert parse_timestamps(pd.Series([1.5, 2.5])).tolist() == [1.5, 2.5]


def test_parallel_directed_arcs_and_ids_survive():
    nodes, arcs, _ = frames()
    graph = build_graph(nodes, arcs)
    assert graph.number_of_edges() == 3
    assert set(graph[0][1]) == {10, 11}


def test_geometry_and_candidate_ordering():
    nodes, arcs, track = frames()
    road = build_road_geometry(nodes, arcs)
    candidates = generate_candidates(track, road, radius_m=200, max_candidates=None)
    assert [c.arc_id for c in candidates] == [10, 11]
