from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from .loader import TripData


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Return elapsed seconds without converting numeric values to epoch datetimes."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(float("nan"), index=values.index, dtype="float64")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        result.loc[numeric_mask] = numeric.loc[numeric_mask].where(numeric.loc[numeric_mask].apply(lambda x: pd.notna(x) and pd.api.types.is_number(x) and float(x) == float(x) and abs(float(x)) != float("inf")))
    text_mask = ~numeric_mask
    if text_mask.any():
        parsed = pd.to_datetime(values.loc[text_mask], errors="coerce", utc=True)
        result.loc[text_mask] = parsed.astype("int64", errors="ignore") / 1_000_000_000
    return result


def integer_ids(values: pd.Series, field: str, report: "ValidationReport") -> set[int]:
    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.isna() | (numeric % 1 != 0)
    if bad.any():
        rows = [int(i) for i in values.index[bad][:5]]
        report.errors.append(f"{field} contains non-integer IDs at rows {rows}")
        return set()
    return set(numeric.astype("int64"))


@dataclass
class ValidationReport:
    trip_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_trip(trip: TripData) -> ValidationReport:
    r = ValidationReport(trip.trip_id)
    for name, frame in (("track", trip.track), ("nodes", trip.nodes), ("arcs", trip.arcs), ("route", trip.route)):
        if frame.empty:
            r.errors.append(f"{name} is empty")
        if frame.isna().any().any():
            r.errors.append(f"{name} contains missing values")
    for name, frame in (("track", trip.track), ("nodes", trip.nodes)):
        if not frame.empty and ((~pd.to_numeric(frame.longitude, errors="coerce").between(-180, 180)) | (~pd.to_numeric(frame.latitude, errors="coerce").between(-90, 90))).any():
            r.errors.append(f"{name} contains invalid coordinates")
    if not trip.track.empty:
        times = parse_timestamps(trip.track.timestamp)
        if times.isna().any() or (~times.apply(pd.api.types.is_number)).any(): r.errors.append("track contains invalid or non-finite timestamps")
        if not times.is_monotonic_increasing: r.errors.append("track timestamps are not ordered")
        if trip.track.duplicated().any(): r.warnings.append("track contains duplicate observations")
    node_ids = integer_ids(trip.nodes.node_id, "node_id", r)
    if not trip.arcs.empty:
        from_ids = integer_ids(trip.arcs.from_node_id, "from_node_id", r)
        to_ids = integer_ids(trip.arcs.to_node_id, "to_node_id", r)
        refs = from_ids | to_ids
        if not refs.issubset(node_ids): r.errors.append("arcs reference nodes outside the zero-based node table")
    arc_ids = integer_ids(trip.arcs.arc_id, "arc_id", r)
    if not trip.route.empty:
        route_ids = integer_ids(trip.route.arc_id, "route.arc_id", r)
        if not route_ids.issubset(arc_ids): r.errors.append("route references arcs outside the zero-based arc table")
    return r
