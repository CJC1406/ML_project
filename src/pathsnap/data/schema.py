from dataclasses import dataclass
from pathlib import Path
import csv


class DatasetError(ValueError):
    """Raised when an input trip is incomplete or malformed."""


@dataclass(frozen=True)
class TripFiles:
    track: Path
    nodes: Path
    arcs: Path
    route: Path | None = None


REQUIRED = {
    "track": {"time", "latitude", "longitude"},
    "nodes": {"node_id", "latitude", "longitude"},
    "arcs": {"arc_id", "from_node", "to_node"},
}


def _header(path: Path) -> set[str]:
    if not path.is_file():
        raise DatasetError(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.reader(fh), None)
    if not row:
        raise DatasetError(f"File has no header: {path}")
    return {x.strip().lower() for x in row}


def validate_trip(files: TripFiles) -> None:
    """Validate only the stable semantic contract, not unseen dataset details."""
    for name, path in (("track", files.track), ("nodes", files.nodes), ("arcs", files.arcs)):
        missing = REQUIRED[name] - _header(path)
        if missing:
            raise DatasetError(f"{name} is missing columns: {sorted(missing)}")
    if files.route is not None and not files.route.is_file():
        raise DatasetError(f"Missing route file: {files.route}")
