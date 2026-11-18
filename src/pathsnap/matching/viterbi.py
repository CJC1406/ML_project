from math import log


def viterbi(emissions: list[dict[str, float]], transitions: dict[tuple[str, str], float]) -> list[str]:
    """Return the best path over changing candidate sets in log space.

    Zero-probability emissions/transitions are impossible (negative infinity),
    never replaced by a tiny positive probability. ``ValueError`` means no
    complete valid path exists.
    """
    if not emissions or any(not row for row in emissions):
        raise ValueError("emissions must contain at least one non-empty observation")
    states = [sorted(row) for row in emissions]
    def lp(value: float) -> float:
        return log(value) if value > 0 else float("-inf")
    score = {s: lp(float(emissions[0][s])) for s in states[0]}
    back: list[dict[str, str | None]] = [{s: None for s in states[0]}]
    for row in emissions[1:]:
        nxt, pointers = {}, {}
        for state in sorted(row):
            best_prev, best_score = None, float("-inf")
            for prev, prev_score in score.items():
                candidate = prev_score + lp(float(transitions.get((prev, state), 0.0)))
                if candidate > best_score:
                    best_prev, best_score = prev, candidate
            nxt[state] = best_score + lp(float(row[state]))
            pointers[state] = best_prev
        score, back = nxt, back + [pointers]
        if not score or max(score.values()) == float("-inf"):
            raise ValueError("no valid Viterbi path exists for the supplied candidate sets")
    current = max(score, key=score.get)
    path = [current]
    for pointers in reversed(back[1:]):
        current = pointers[current]
        path.append(current)
    return list(reversed(path))
