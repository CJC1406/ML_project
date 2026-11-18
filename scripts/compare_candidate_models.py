"""Compare candidate scorers on the supplied train and student-test partitions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import shapely
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from pathsnap.data.loader import discover_trip_directories, load_trip
from pathsnap.data.validator import validate_trip
from pathsnap.geo.projection import choose_trip_projection

FEATURES = ["distance_m", "heading_difference_deg", "speed_mps", "gap_seconds",
            "candidate_rank", "distance_above_nearest_m"]
SEED = 2026


def nearby(points, lines, radius):
    pairs = shapely.STRtree(lines).query(points, predicate="dwithin", distance=radius)
    observation, arc = pairs
    distance = shapely.distance(points[observation], lines[arc])
    ordered = np.lexsort((arc, distance, observation))
    counts = np.bincount(observation, minlength=len(points))
    return np.split(arc[ordered], np.cumsum(counts)[:-1]), np.split(
        distance[ordered], np.cumsum(counts)[:-1])


def align(reference, road_ends, positions, distances, gaps):
    """Decode ordered reference occurrences, preserving repeat visits to an arc."""
    labels = np.full(len(positions), -1, dtype=int)
    breaks = np.r_[False, road_ends[reference[:-1], 1] != road_ends[reference[1:], 0]]
    component = np.cumsum(breaks)
    # Each run ends at an observation gap, a missing reference candidate, or
    # when there is no forward transition within a continuous reference block.
    start = 0
    while start < len(positions):
        if not len(positions[start]):
            start += 1
            continue
        end = start + 1
        forward = [distances[start].copy()]
        transitions = []
        while end < len(positions) and gaps[end] <= 60 and len(positions[end]):
            previous, current = positions[end - 1], positions[end]
            steps = current[None, :] - previous[:, None]
            allowed = (steps >= 0) & (component[previous, None] == component[current][None, :])
            penalty = np.where(allowed, 2.0 * np.maximum(steps - 1, 0), np.inf)
            cost = np.min(forward[-1][:, None] + penalty, axis=0) + distances[end]
            if not np.any(np.isfinite(cost)):
                break
            transitions.append(penalty)
            forward.append(cost)
            end += 1

        backward = np.zeros(len(positions[end - 1]))
        for obs in range(end - 1, start - 1, -1):
            cost = forward[obs - start] + backward
            best = int(np.argmin(cost))
            arc = reference[positions[obs][best]]
            alternatives = cost[reference[positions[obs]] != arc]
            if not alternatives.size or alternatives.min() - cost[best] >= 5.0:
                labels[obs] = arc
            if obs > start:
                backward = np.min(
                    transitions[obs - start - 1] + (distances[obs] + backward)[None, :],
                    axis=1,
                )
        start = end
    return labels


def trip_rows(directory):
    trip = load_trip(directory)
    report = validate_trip(trip)
    if not report.valid:
        raise ValueError(f"{trip.trip_id}: {report.errors}")
    road_ends = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    reference = trip.route.arc_id.to_numpy(dtype=int)
    projection = choose_trip_projection(trip.nodes.longitude, trip.nodes.latitude)

    def xy(frame):
        return np.column_stack(projection.forward.transform(
            frame.longitude.to_numpy(), frame.latitude.to_numpy()))

    node_xy, track_xy = xy(trip.nodes), xy(trip.track)
    points = shapely.points(track_xy)
    lines = shapely.linestrings(node_xy[road_ends])
    delta = np.diff(track_xy, axis=0, prepend=track_xy[:1])
    movement = np.linalg.norm(delta, axis=1)
    gaps = np.diff(trip.track.timestamp.to_numpy(dtype=float), prepend=np.nan)
    speed = np.divide(movement, gaps, out=np.full(len(points), np.nan),
                      where=(gaps > 0) & (gaps <= 60))
    heading = np.degrees(np.arctan2(delta[:, 0], delta[:, 1]))
    heading[~((gaps > 0) & (gaps <= 60) & (movement >= 1))] = np.nan
    road_delta = node_xy[road_ends[:, 1]] - node_xy[road_ends[:, 0]]
    road_heading = np.degrees(np.arctan2(road_delta[:, 0], road_delta[:, 1]))

    # Reference-only label alignment is separate from inference candidates.
    positions, reference_distances = nearby(points, lines[reference], radius=30)
    labels = align(reference, road_ends, positions, reference_distances, gaps)
    arcs, distances = nearby(points, lines, radius=50)
    examples = []
    stats = dict(observations=len(points), aligned=int((labels >= 0).sum()),
                 with_candidates=sum(bool(len(a)) for a in arcs),
                 aligned_covered=0, candidate_rows=sum(min(8, len(a)) for a in arcs))
    for obs, (gold, arc, distance) in enumerate(zip(labels, arcs, distances)):
        if gold < 0 or not len(arc):
            continue
        arc, distance = arc[:8], distance[:8]
        target = arc == gold
        if not target.any():
            continue
        stats["aligned_covered"] += 1
        difference = np.abs((heading[obs] - road_heading[arc] + 180) % 360 - 180)
        rows = np.column_stack((distance, difference, np.full(len(arc), speed[obs]),
                                np.full(len(arc), gaps[obs]), np.arange(len(arc)),
                                distance - distance[0], target))
        examples.append(rows)
    return examples, stats


def load_partition(root):
    examples, groups, trip_ids = [], [], set()
    totals = dict(observations=0, aligned=0, with_candidates=0, aligned_covered=0,
                  candidate_rows=0)
    for directory in discover_trip_directories(root):
        rows, stats = trip_rows(directory)
        trip_ids.add(directory.name)
        examples.extend(rows)
        groups.extend([directory.name] * len(rows))
        for key, value in stats.items():
            totals[key] += value
        if len(trip_ids) % 10 == 0:
            print(f"{root.name}: processed {len(trip_ids)} trips", flush=True)
    return examples, np.asarray(groups), trip_ids, totals


def stack(examples, indices):
    selected = [examples[i] for i in indices]
    rows = np.concatenate(selected)
    offsets = np.r_[0, np.cumsum([len(row) for row in selected])]
    return rows[:, :-1], rows[:, -1].astype(int), offsets


def score(probabilities, y, offsets, aligned_count=None):
    winners = [first + np.argmax(probabilities[first:last])
               for first, last in zip(offsets[:-1], offsets[1:])]
    correct = int(y[winners].sum())
    result = dict(top1=correct / len(winners), correct=correct, scored=len(winners),
                  candidate_f1=float(f1_score(y, probabilities >= 0.5, zero_division=0)),
                  candidate_log_loss=float(log_loss(y, probabilities, labels=[0, 1])))
    if aligned_count:
        result["top1_all_aligned"] = correct / aligned_count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train, groups, train_ids, train_stats = load_partition(args.dataset / "train")
    test, _, test_ids, test_stats = load_partition(args.dataset / "student_test")
    if (len(train_ids), len(test_ids)) != (80, 10) or train_ids & test_ids:
        raise ValueError("Expected 80 training and 10 distinct student-test trips.")
    fit, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                           .split(groups, groups=groups))
    fit_x, fit_y, _ = stack(train, fit)
    val_x, val_y, val_offsets = stack(train, validation)
    train_x, train_y, _ = stack(train, range(len(train)))
    test_x, test_y, test_offsets = stack(test, range(len(test)))
    results = dict(seed=SEED, features=FEATURES,
                   fit_trips=len(set(groups[fit])), validation_trips=len(set(groups[validation])),
                   train=train_stats, student_test=test_stats, models={})
    results["nearest_road"] = {
        "validation_top1": float(val_y[val_offsets[:-1]].mean()),
        "student_test_top1": float(test_y[test_offsets[:-1]].mean()),
        "student_test_top1_all_aligned": int(test_y[test_offsets[:-1]].sum()) / test_stats["aligned"],
    }
    models = {
        "decision_tree": make_pipeline(SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=30, random_state=SEED)),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, early_stopping=False,
            random_state=SEED),
    }
    with threadpool_limits(limits=4):
        for name, model in models.items():
            model.fit(fit_x, fit_y)
            val_result = score(model.predict_proba(val_x)[:, 1], val_y, val_offsets)
            start = time.perf_counter()
            model.fit(train_x, train_y)
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            probabilities = model.predict_proba(test_x)[:, 1]
            inference_time = time.perf_counter() - start
            results["models"][name] = dict(
                validation=val_result,
                student_test=score(probabilities, test_y, test_offsets, test_stats["aligned"]),
                fit_seconds=fit_time, test_scoring_seconds=inference_time)
            if name == "decision_tree":
                results["models"][name]["feature_importances"] = dict(zip(
                    FEATURES, model[-1].feature_importances_.tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
