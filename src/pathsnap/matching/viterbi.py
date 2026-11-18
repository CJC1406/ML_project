from math import log


def viterbi(emissions: list[dict[str, float]], transitions: dict[tuple[str, str], float]) -> list[str]:
    """Return the highest-probability state sequence using log probabilities."""
    if not emissions or any(not row for row in emissions):
        raise ValueError("emissions must contain at least one non-empty observation")
    states = sorted({state for row in emissions for state in row})
    score = {s: log(max(emissions[0].get(s, 0.0), 1e-300)) for s in states}
    back: list[dict[str, str | None]] = [{s: None for s in states}]
    for row in emissions[1:]:
        nxt, pointers = {}, {}
        for state in states:
            best_prev, best_score = None, float("-inf")
            for prev in states:
                transition = transitions.get((prev, state), 0.0)
                candidate = score[prev] + log(max(transition, 1e-300))
                if candidate > best_score:
                    best_prev, best_score = prev, candidate
            nxt[state] = best_score + log(max(row.get(state, 0.0), 1e-300))
            pointers[state] = best_prev
        score, back = nxt, back + [pointers]
    current = max(score, key=score.get)
    path = [current]
    for pointers in reversed(back[1:]):
        current = pointers[current]
        path.append(current)
    return list(reversed(path))
