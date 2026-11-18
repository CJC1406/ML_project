"""Sequence-level performance of the KNN classifier on its own (no HMM), on the student-testing trips.

Predicted route = for every GPS observation take the highest-probability candidate arc, then collapse
consecutive repeats. Compared against the reference .route (backend-only):
  * route precision  = length of predicted arcs that are in the reference route / predicted length
  * route recall     = length of reference arcs that were predicted / reference length
  * route F1         = harmonic mean of the two
  * route mismatch fraction (Newson & Krumm 2009) = (wrongly added length + missed length) / reference length
  * route continuity = share of consecutive predicted arcs that are connected (to-node == next from-node)
All arc-set metrics are length-weighted (metres), so a long missed motorway arc counts more than a 3 m stub.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from pathsnap.features.candidate_features import load_cache, load_trip, project


def route_metrics(pred_seq, route, arc_len):
    P, G = set(pred_seq.tolist()), set(route.tolist())
    lp, lg = arc_len[list(P)].sum(), arc_len[list(G)].sum()
    li = arc_len[list(P & G)].sum() if P & G else 0.0
    prec, rec = li / max(lp, 1e-9), li / max(lg, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    added = arc_len[list(P - G)].sum() if P - G else 0.0
    missed = arc_len[list(G - P)].sum() if G - P else 0.0
    return prec, rec, f1, (added + missed) / max(lg, 1e-9)


def main(data_root, cache_dir, res_dir, split="student_test"):
    probs = np.load(os.path.join(res_dir, "knn_student_test_probs.npz"))
    rows = []
    for tr in load_cache(cache_dir, split):
        tid = str(tr["trip"])
        _, nodes, arcs, track, route = load_trip(os.path.join(data_root, split, tid))
        P = project(nodes[:, 0], nodes[:, 1], track[:, 0].mean(), track[:, 1].mean())
        arc_len = np.hypot(*(P[arcs[:, 1]] - P[arcs[:, 0]]).T)
        p, obs, cand = probs[tid], tr["obs"], tr["cand"]
        order = np.lexsort((-p, obs))
        first = np.r_[True, obs[order][1:] != obs[order][:-1]]
        seq = cand[order][first]                       # best arc per observation, in time order
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]   # collapse repeats
        prec, rec, f1, rmf = route_metrics(seq, route, arc_len)
        cont = float(np.mean(arcs[seq[:-1], 1] == arcs[seq[1:], 0])) if len(seq) > 1 else 1.0
        rows.append(dict(trip=tid, route_precision=prec, route_recall=rec, route_f1=f1,
                         route_mismatch_fraction=rmf, route_continuity=cont,
                         predicted_arcs=int(len(seq)), reference_arcs=int(len(route))))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "trip"}
    out = dict(model="KNN (classifier only, no HMM)", split=split, per_trip=rows, mean=mean)
    json.dump(out, open(os.path.join(res_dir, "knn_sequence_metrics.json"), "w"), indent=2, default=float)
    print(f"{'trip':>10} {'prec':>7} {'recall':>7} {'F1':>7} {'mismatch':>9} {'contin.':>8}")
    for r in rows:
        print(f"{r['trip']:>10} {r['route_precision']:7.3f} {r['route_recall']:7.3f} {r['route_f1']:7.3f} "
              f"{r['route_mismatch_fraction']:9.3f} {r['route_continuity']:8.3f}")
    print(f"{'mean':>10} {mean['route_precision']:7.3f} {mean['route_recall']:7.3f} {mean['route_f1']:7.3f} "
          f"{mean['route_mismatch_fraction']:9.3f} {mean['route_continuity']:8.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
