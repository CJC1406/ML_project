from dataclasses import dataclass, field
import pandas as pd
from .loader import TripData


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
        times = pd.to_datetime(trip.track.timestamp, errors="coerce", utc=True)
        if times.isna().any(): r.errors.append("track contains invalid timestamps")
        if not times.is_monotonic_increasing: r.errors.append("track timestamps are not ordered")
        if trip.track.duplicated().any(): r.warnings.append("track contains duplicate observations")
    node_ids = set(range(len(trip.nodes)))
    if not trip.arcs.empty:
        refs = set(pd.to_numeric(trip.arcs.from_node_id, errors="coerce").dropna().astype(int)) | set(pd.to_numeric(trip.arcs.to_node_id, errors="coerce").dropna().astype(int))
        if not refs.issubset(node_ids): r.errors.append("arcs reference nodes outside the zero-based node table")
    arc_ids = set(range(len(trip.arcs)))
    if not trip.route.empty:
        route_ids = set(pd.to_numeric(trip.route.arc_id, errors="coerce").dropna().astype(int))
        if not route_ids.issubset(arc_ids): r.errors.append("route references arcs outside the zero-based arc table")
    return r
