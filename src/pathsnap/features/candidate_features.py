"""PathSnap shared data pipeline: load trip -> project to metres -> candidate roads -> labels -> features.

One row = (GPS observation i, candidate arc a). Label y = 1 if a is the arc the vehicle was on at i.

Design decisions (explain these in the viva / report):
* Distances: each trip is projected to a local equirectangular plane (metres) centred on the trip.
  Trips span < ~80 km, where this projection's distance error is well under 1 %, and it lets us use
  fast Euclidean point-to-segment geometry instead of haversine on every pair.
* Candidates: every arc whose segment lies within RADIUS_M of the GPS point, keeping the K nearest.
  Both directions of a two-way road are separate arcs, so direction is decided by features.
* Labels: the .route file lists arcs in driving order but does NOT say which GPS point belongs to
  which arc. We align points to route arcs with a monotone dynamic programme (point index and route
  index both move forward; cost = point-to-arc distance). A point whose aligned arc is more than
  RADIUS_M away is treated as unlabelled (reference route missing/offset there) and excluded from
  training/metrics - see Data Truths on trips 2, 33, 54.
* Features use only the GPS track and the map (never the route), so they are valid at prediction time.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371008.8
RADIUS_M = 50.0        # candidate search radius (p95 GPS error ~14 m, p99 < 50 m on clean trips)
MAX_CAND = 10          # keep at most K nearest candidate arcs per observation
DENSIFY_M = 10.0       # spacing of helper points along arcs for the KD-tree
WIN = 3                # +/- window of neighbouring observations for smoothed-distance feature

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "win_dist", "cos_heading", "heading_ok",
    "speed", "dt_prev", "arc_len", "proj_t", "n_cand", "junction_deg", "has_reverse",
    "cont_prev", "cont_next",
]


# ----------------------------------------------------------------------------- loading
def load_trip(trip_dir: str):
    tid = os.path.basename(os.path.normpath(trip_dir))
    nodes = np.loadtxt(os.path.join(trip_dir, f"{tid}.nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(trip_dir, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
    track = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
    rp = os.path.join(trip_dir, f"{tid}.route")
    route = np.loadtxt(rp, dtype=np.int64, ndmin=1) if os.path.exists(rp) else None
    return tid, nodes, arcs, track, route


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres."""
    k = R_EARTH * np.pi / 180.0
    return np.c_[(lon - lon0) * k * np.cos(np.radians(lat0)), (lat - lat0) * k]


def point_seg(px, py, ax, ay, bx, by):
    """Distance (m) from points to segments, plus projection parameter t in [0,1]. Broadcasts."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(L2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


# ----------------------------------------------------------------------------- labels
def align_route(T, A, B, route):
    """Monotone alignment of GPS points to route arcs. Returns (arc id per point, distance)."""
    ra = route
    ax, ay, bx, by = A[ra, 0], A[ra, 1], B[ra, 0], B[ra, 1]
    N, M = len(T), len(ra)
    prev = None
    back = np.empty((N, M), dtype=np.int32)       # argmin pointer into previous row
    D = np.empty((N, M), dtype=np.float32)
    for s in range(0, N, 512):
        e = min(N, s + 512)
        d, _ = point_seg(T[s:e, 0:1], T[s:e, 1:2], ax[None], ay[None], bx[None], by[None])
        D[s:e] = d
    for i in range(N):
        if prev is None:
            cur = D[0].astype(np.float64)
            back[0] = np.arange(M)
        else:
            # best previous state with route index <= j  (running minimum + its argmin)
            run_min = np.minimum.accumulate(prev)
            idx = np.arange(M)
            is_new = np.r_[True, prev[1:] < run_min[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = D[i] + run_min
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    arc = ra[path]
    dist = D[np.arange(N), path]
    return arc, dist


# ----------------------------------------------------------------------------- features
def build_trip(trip_dir: str, with_labels: bool = True):
    tid, nodes, arcs, track, route = load_trip(trip_dir)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    P = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    T = project(track[:, 0], track[:, 1], lon0, lat0)
    t = track[:, 2]
    N = len(T)
    A, B = P[arcs[:, 0]], P[arcs[:, 1]]
    arc_vec = B - A
    arc_len = np.hypot(arc_vec[:, 0], arc_vec[:, 1])
    arc_dir = arc_vec / np.maximum(arc_len[:, None], 1e-9)

    # --- graph helpers
    n_nodes = len(nodes)
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
    in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
    pair = arcs[:, 0] * n_nodes + arcs[:, 1]
    rev_pair = arcs[:, 1] * n_nodes + arcs[:, 0]
    has_rev = np.isin(rev_pair, pair).astype(np.float32)

    # --- GPS kinematics (central differences; heading unreliable when barely moving)
    dt_prev = np.r_[1.0, np.diff(t)]
    i0 = np.r_[0, np.arange(N - 1)]
    i1 = np.r_[np.arange(1, N), N - 1]
    disp = T[i1] - T[i0]
    disp_len = np.hypot(disp[:, 0], disp[:, 1])
    speed = disp_len / np.maximum(t[i1] - t[i0], 1e-9)
    head = disp / np.maximum(disp_len[:, None], 1e-9)
    heading_ok = (disp_len > 2.0).astype(np.float32)

    # --- candidates via KD-tree on densified arc points
    k = np.maximum(1, np.ceil(arc_len / DENSIFY_M).astype(int))
    owner = np.repeat(np.arange(len(arcs)), k + 1)
    frac = np.concatenate([np.linspace(0, 1, kk + 1) for kk in k])
    helper = A[owner] + arc_vec[owner] * frac[:, None]
    tree = cKDTree(helper)
    _, nn = tree.query(T, k=128, distance_upper_bound=RADIUS_M + DENSIFY_M)
    valid = nn < len(helper)
    cand_lists = []
    for i in range(N):
        a = np.unique(owner[nn[i][valid[i]]])
        if len(a) == 0:
            cand_lists.append(a)
            continue
        d, _ = point_seg(T[i, 0], T[i, 1], A[a, 0], A[a, 1], B[a, 0], B[a, 1])
        keep = d <= RADIUS_M
        a, d = a[keep], d[keep]
        cand_lists.append(a[np.argsort(d, kind="stable")[:MAX_CAND]])
    obs = np.concatenate([np.full(len(c), i) for i, c in enumerate(cand_lists)]).astype(np.int64)
    cand = np.concatenate(cand_lists).astype(np.int64)
    n_cand = np.array([len(c) for c in cand_lists])

    # --- geometric features
    dist, proj_t = point_seg(T[obs, 0], T[obs, 1], A[cand, 0], A[cand, 1], B[cand, 0], B[cand, 1])
    starts = np.r_[0, np.cumsum(n_cand)[:-1]]
    rank = np.arange(len(obs)) - starts[obs]
    min_d =np.array([dist[s:s + n].min() if n else np.nan for s, n in zip(starts, n_cand)])
    nearest_arc = np.array([cand[s] if n else -1 for s, n in zip(starts, n_cand)])

    # smoothed distance: mean distance of neighbouring observations (i-WIN..i+WIN) to this arc
    acc = np.zeros(len(obs))
    for o in range(-WIN, WIN + 1):
        j = np.clip(obs + o, 0, N - 1)
        dd, _ = point_seg(T[j, 0], T[j, 1], A[cand, 0], A[cand, 1], B[cand, 0], B[cand, 1])
        acc += dd
    win_dist = acc / (2 * WIN + 1)

    cos_h = (head[obs] * arc_dir[cand]).sum(1) * heading_ok[obs]

    # continuity with neighbours' nearest arc (uses only GPS+map, no labels)
    def cont(other_obs, forward):
        oa = nearest_arc[other_obs]
        same = (oa == cand)
        if forward:   # previous arc ends where candidate starts
            adj = (oa >= 0) & (arcs[np.maximum(oa, 0), 1] == arcs[cand, 0])
        else:         # candidate ends where next arc starts
            adj = (oa >= 0) & (arcs[cand, 1] == arcs[np.maximum(oa, 0), 0])
        return np.where(same, 1.0, np.where(adj, 0.5, 0.0))
    cont_prev = cont(np.maximum(obs - 1, 0), True)
    cont_next = cont(np.minimum(obs + 1, N - 1), False)

    X = np.c_[
        dist, rank, dist - min_d[obs], win_dist, cos_h, heading_ok[obs],
        speed[obs], np.minimum(dt_prev[obs], 600.0), arc_len[cand], proj_t, n_cand[obs],
        in_deg[arcs[cand, 0]] + out_deg[arcs[cand, 1]], has_rev[cand], cont_prev, cont_next,
    ].astype(np.float32)

    out = dict(trip=tid, X=X, obs=obs, cand=cand, n_obs=N, n_cand=n_cand)
    if with_labels and route is not None:
        true_arc, true_d = align_route(T, A, B, route)
        labelled = true_d <= RADIUS_M
        out["true_arc"] = np.where(labelled, true_arc, -1)
        out["y"] = ((cand == true_arc[obs]) & labelled[obs]).astype(np.int8)
        out["row_labelled"] = labelled[obs]
    return out


def cache_all(root: str, cache_dir: str, splits=("train", "student_test")):
    """Build features for every trip once and store as .npz (teammates can reuse these)."""
    import time
    os.makedirs(cache_dir, exist_ok=True)
    for split in splits:
        for tid in sorted(os.listdir(os.path.join(root, split))):
            f = os.path.join(cache_dir, f"{split}_{tid}.npz")
            if os.path.exists(f):
                continue
            s = time.time()
            d = build_trip(os.path.join(root, split, tid))
            np.savez_compressed(f, split=split, **d)
            print(f"{split}/{tid}: {d['n_obs']} obs, {len(d['obs'])} rows, {time.time()-s:.1f}s", flush=True)


def load_cache(cache_dir: str, split: str):
    trips = []
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith(split + "_") and f.endswith(".npz"):
            z = np.load(os.path.join(cache_dir, f), allow_pickle=False)
            trips.append({k: z[k] for k in z.files})
    return trips


if __name__ == "__main__":
    import sys
    cache_all(sys.argv[1], sys.argv[2])
