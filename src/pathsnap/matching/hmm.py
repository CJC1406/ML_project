"""HMM + Viterbi route reconstruction over classifier-scored candidate arcs.

Hidden states   at GPS observation i: its candidate arcs (same candidates the classifier scored).
Emissions       log e_i(a) = alpha * log P(a | features), with P normalised over i's candidates.
                P comes from the trained candidate classifier, so the HMM inherits its evidence.
Transitions     Newson & Krumm (2009): log T(a -> b) = -|d_road(a, b) - d_gps(i, i+1)| / beta, where
                d_road is the shortest driving distance from i's projected position on a to i+1's
                projected position on b, and d_gps the straight-line distance between the two fixes.
                * staying on an arc moves forward along it; small backward GPS jitter costs nothing
                * b unreachable within the search limit -> transition impossible (-inf)
Breaks          the chain restarts (a new Viterbi segment) when an observation has no candidates,
                the time gap exceeds MAX_GAP_S, or no finite transition exists. Breaks are reported.
Reconstruction  best arc per observation -> collapse repeats -> join consecutive arcs that do not
                touch with the shortest network path, so the output is a connected arc sequence
                wherever the network allows it.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from pathsnap.features.candidate_features import FEATURES, project

MAX_GAP_S = 60.0
PROJ_T = FEATURES.index("proj_t")


class RoadGraph:
    """Directed road graph with lazily computed, distance-limited shortest paths between nodes."""

    def __init__(self, nodes_lonlat, arcs, lon0, lat0):
        self.arcs = np.asarray(arcs, dtype=np.int64)[:, :2]
        xy = project(nodes_lonlat[:, 0], nodes_lonlat[:, 1], lon0, lat0)
        d = xy[self.arcs[:, 1]] - xy[self.arcs[:, 0]]
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.out = [[] for _ in range(len(nodes_lonlat))]
        for arc_id, (u, v) in enumerate(self.arcs):
            self.out[u].append((v, arc_id))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def from_node(self, src: int, limit: float):
        """Shortest distances (and the arc used to reach each node) from src, up to `limit` metres."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        dist, via = {src: 0.0}, {src: -1}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for v, arc_id in self.out[u]:
                nd = d + self.length[arc_id]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v], via[v] = nd, arc_id
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist, via)
        return dist, via

    def road_distance(self, a, ta, b, tb, limit):
        """Driving distance from fraction ta along arc a to fraction tb along arc b (inf if > limit)."""
        if a == b:
            return max(0.0, tb - ta) * self.length[a]
        rest = (1.0 - ta) * self.length[a]
        dist, _ = self.from_node(int(self.arcs[a, 1]), limit)
        mid = dist.get(int(self.arcs[b, 0]))
        if mid is None or mid > limit:          # the cache may hold a search with a larger limit
            return np.inf
        return rest + mid + tb * self.length[b]

    def connecting_arcs(self, a, b, limit):
        """Arcs strictly between a and b on the shortest path end(a) -> start(b), or None."""
        src, dst = int(self.arcs[a, 1]), int(self.arcs[b, 0])
        if src == dst:
            return []
        dist, via = self.from_node(src, limit)
        if dst not in dist:
            return None
        path, node = [], dst
        while node != src:
            arc_id = via[node]
            path.append(arc_id)
            node = int(self.arcs[arc_id, 0])
        return path[::-1]


@dataclass
class TripLattice:
    """Everything Viterbi needs for one trip, independent of alpha and beta (so tuning is cheap)."""
    n_obs: int
    states: list            # per observation: candidate arc ids (np.ndarray), maybe empty
    log_p: list             # per observation: log normalised classifier probability per candidate
    mismatch: list          # per step i-1 -> i: |d_road - d_gps| matrix (inf = impossible), or None at a break
    graph: RoadGraph = field(repr=False)


def build_lattice(trip_cache: dict, probs, nodes_lonlat, arcs, track) -> TripLattice:
    obs, cand, X = trip_cache["obs"], trip_cache["cand"], trip_cache["X"]
    n = int(trip_cache["n_obs"])
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    graph = RoadGraph(nodes_lonlat, arcs, lon0, lat0)
    gps = project(track[:, 0], track[:, 1], lon0, lat0)
    t = track[:, 2]
    starts = np.searchsorted(obs, np.arange(n + 1))
    states, log_p, frac = [], [], []
    for i in range(n):
        s, e = starts[i], starts[i + 1]
        p = np.clip(np.asarray(probs[s:e], float), 1e-9, None)
        states.append(cand[s:e]); log_p.append(np.log(p / p.sum()) if e > s else p)
        frac.append(X[s:e, PROJ_T].astype(float))
    mismatch = [None]
    for i in range(1, n):
        if not len(states[i]) or not len(states[i - 1]) or t[i] - t[i - 1] > MAX_GAP_S:
            mismatch.append(None)
            continue
        d_gps = float(np.hypot(*(gps[i] - gps[i - 1])))
        limit = 2.0 * d_gps + 200.0
        m = np.empty((len(states[i - 1]), len(states[i])))
        for r, (a, ta) in enumerate(zip(states[i - 1], frac[i - 1])):
            for c, (b, tb) in enumerate(zip(states[i], frac[i])):
                m[r, c] = abs(graph.road_distance(int(a), ta, int(b), tb, limit) - d_gps)
        mismatch.append(m)
    return TripLattice(n, states, log_p, mismatch, graph)


def decode(lat: TripLattice, alpha: float = 1.0, beta: float = 10.0):
    """Viterbi in log space with chain restarts. Returns (arc per observation or -1, number of breaks)."""
    pred = np.full(lat.n_obs, -1, dtype=np.int64)
    breaks = 0
    seg_start, score, back = None, None, []

    def flush(end):
        if seg_start is None:
            return
        j = int(np.argmax(score))
        for i in range(end, seg_start - 1, -1):
            pred[i] = lat.states[i][j]
            if i > seg_start:
                j = back[i - seg_start - 1][j]

    for i in range(lat.n_obs):
        emit = alpha * lat.log_p[i] if len(lat.states[i]) else None
        m = lat.mismatch[i] if i else None
        if emit is None:
            flush(i - 1); seg_start, score, back = None, None, []
            continue
        if seg_start is not None and m is not None:
            total = score[:, None] - m / beta
            best = np.argmax(total, axis=0)
            new = total[best, np.arange(total.shape[1])]
            if np.isfinite(new).any():
                back.append(best); score = new + emit
                continue
        if seg_start is not None:
            breaks += 1
        flush(i - 1)
        seg_start, score, back = i, emit.copy(), []
    flush(lat.n_obs - 1)
    return pred, breaks


def classifier_only(lat: TripLattice):
    return np.array([s[int(np.argmax(p))] if len(s) else -1 for s, p in zip(lat.states, lat.log_p)])


def reconstruct_route(graph: RoadGraph, pred, fill_limit: float = 2000.0):
    """Collapse per-observation arcs into a route; join non-touching neighbours by shortest path."""
    seq = pred[pred >= 0]
    if not len(seq):
        return np.array([], dtype=np.int64), 0
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    route, unjoined = [int(seq[0])], 0
    for b in seq[1:]:
        a = route[-1]
        if graph.arcs[a, 1] != graph.arcs[b, 0]:
            link = graph.connecting_arcs(a, int(b), fill_limit)
            if link is None:
                unjoined += 1
            else:
                route.extend(link)
        route.append(int(b))
    route = np.array(route, dtype=np.int64)
    return route[np.r_[True, route[1:] != route[:-1]]], unjoined
