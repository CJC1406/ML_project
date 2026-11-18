from dataclasses import dataclass, field
import pandas as pd
from .loader import TripData


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse numeric timestamps as Unix seconds; textual values use pandas parsing."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_datetime(numeric.loc[numeric_mask], unit="s", errors="coerce", utc=True)
    text_mask = ~numeric_mask
    if text_mask.any():
        result.loc[text_mask] = pd.to_datetime(values.loc[text_mask], errors="coerce", utc=True)
    return result


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
        if times.isna().any(): r.errors.append("track contains invalid timestamps")
        if not times.is_monotonic_increasing: r.errors.append("track timestamps are not ordered")
        if trip.track.duplicated().any(): r.warnings.append("track contains duplicate observations")
    node_ids = set(pd.to_numeric(trip.nodes.node_id, errors="coerce").dropna().astype(int))
    if not trip.arcs.empty:
        refs = set(pd.to_numeric(trip.arcs.from_node_id, errors="coerce").dropna().astype(int)) | set(pd.to_numeric(trip.arcs.to_node_id, errors="coerce").dropna().astype(int))
        if not refs.issubset(node_ids): r.errors.append("arcs reference nodes outside the zero-based node table")
    arc_ids = set(pd.to_numeric(trip.arcs.arc_id, errors="coerce").dropna().astype(int))
    if not trip.route.empty:
        route_ids = set(pd.to_numeric(trip.route.arc_id, errors="coerce").dropna().astype(int))
        if not route_ids.issubset(arc_ids): r.errors.append("route references arcs outside the zero-based arc table")
    return r
