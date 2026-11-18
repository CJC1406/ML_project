"""HMM/Viterbi route reconstruction on a tiny hand-made network (no dataset needed)."""
import numpy as np

from pathsnap.features.candidate_features import FEATURES
from pathsnap.matching.hmm import (RoadGraph, TripLattice, build_lattice, classifier_only, decode,
                                   reconstruct_route)

# Straight east-west road at the equator: nodes 0-1-2-3 about 111 m apart, arcs both directions.
NODES = np.array([[0.000, 0.0], [0.001, 0.0], [0.002, 0.0], [0.003, 0.0]])
ARCS = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])   # 0-2 eastbound, 3-5 westbound


def graph():
    return RoadGraph(NODES, ARCS, 0.0015, 0.0)


def test_road_distance_forward_and_unreachable():
    g = graph()
    assert abs(g.road_distance(0, 0.5, 1, 0.5, 1000) - g.length[0]) < 1e-6
    assert g.road_distance(0, 0.2, 0, 0.1, 1000) == 0.0          # backward jitter on one arc is free
    assert g.road_distance(0, 0.5, 2, 0.5, 10) == np.inf         # beyond the search limit


def test_connecting_arcs_fill_gaps():
    g = graph()
    assert g.connecting_arcs(0, 2, 1000) == [1]
    route, unjoined = reconstruct_route(g, np.array([0, 0, 2, 2]))
    assert route.tolist() == [0, 1, 2] and unjoined == 0


def _trip(track, cands, probs):
    obs = np.repeat(np.arange(len(cands)), [len(c) for c in cands])
    X = np.zeros((len(obs), len(FEATURES)), np.float32)
    X[:, FEATURES.index("proj_t")] = 0.5
    return dict(obs=obs, cand=np.concatenate(cands), X=X, n_obs=len(cands)), np.concatenate(probs)


def test_hmm_overrides_an_isolated_classifier_mistake():
    # Vehicle drives east; at the middle reading the classifier prefers the westbound twin arc.
    track = np.array([[0.0005, 0.0, 0], [0.0015, 0.0, 1], [0.0025, 0.0, 2]])
    cache, p = _trip(track, [np.array([0, 3]), np.array([1, 4]), np.array([2, 5])],
                     [np.array([0.9, 0.1]), np.array([0.4, 0.6]), np.array([0.9, 0.1])])
    lat = build_lattice(cache, p, NODES, ARCS, track)
    assert classifier_only(lat).tolist() == [0, 4, 2]
    pred, breaks = decode(lat, alpha=1.0, beta=10.0)
    assert pred.tolist() == [0, 1, 2] and breaks == 0


def test_large_gap_restarts_the_chain():
    track = np.array([[0.0005, 0.0, 0], [0.0015, 0.0, 500]])        # 500 s gap
    cache, p = _trip(track, [np.array([0]), np.array([4])], [np.array([1.0]), np.array([1.0])])
    lat = build_lattice(cache, p, NODES, ARCS, track)
    assert lat.mismatch[1] is None
    pred, breaks = decode(lat)
    assert pred.tolist() == [0, 4] and breaks == 1
