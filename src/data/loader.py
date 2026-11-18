from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


class DatasetLoadError(ValueError):
    """Raised when a trip cannot be loaded safely."""


@dataclass(frozen=True)
class TripData:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame


SCHEMAS = {
    "track": ["longitude", "latitude", "timestamp"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node_id", "to_node_id"],
    "route": ["arc_id"],
}


def _read(path: Path, kind: str) -> pd.DataFrame:
    if not path.is_file():
        raise DatasetLoadError(f"Missing {kind} file: {path}")
    if path.stat().st_size == 0:
        raise DatasetLoadError(f"Empty {kind} file: {path}")
    names = SCHEMAS[kind]
    raw = pd.read_csv(path, sep=r"\s+|,|;|\t", engine="python", header=None, comment="#")
    if raw.empty:
        raise DatasetLoadError(f"Empty {kind} file: {path}")
    first = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
    known = set(names) | {"lat", "lon", "lng", "time", "timestamp"}
    has_header = any(x in known for x in first)
    if has_header:
        raw = pd.read_csv(path, sep=r"\s+|,|;|\t", engine="python", comment="#")
        raw.columns = [str(x).strip().lower() for x in raw.columns]
        aliases = {"lon": "longitude", "lng": "longitude", "lat": "latitude", "time": "timestamp"}
        raw = raw.rename(columns=aliases)
        missing = set(names) - set(raw.columns)
        if missing:
            raise DatasetLoadError(f"{kind} missing columns {sorted(missing)}: {path}")
        return raw[names].copy()
    if raw.shape[1] != len(names):
        raise DatasetLoadError(f"{kind} expected {len(names)} columns, found {raw.shape[1]}: {path}")
    raw.columns = names
    return raw


def discover_trip_directories(root: Path) -> list[Path]:
    """Discover directories containing at least one recognised trip file."""
    if not root.is_dir():
        raise DatasetLoadError(f"Dataset root does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    return sorted(p for p in candidates if any(p.glob(f"*.{kind}") for kind in SCHEMAS))


def load_trip(directory: Path) -> TripData:
    files = {kind: next(directory.glob(f"*.{kind}"), None) for kind in SCHEMAS}
    missing = [kind for kind, path in files.items() if path is None]
    if missing:
        raise DatasetLoadError(f"Trip {directory} missing files: {missing}")
    return TripData(directory.name, *[_read(files[kind], kind) for kind in SCHEMAS])
