"""Compatibility import; implementation lives in pathsnap.data.loader."""
from pathsnap.data.loader import *


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
    raw = pd.read_csv(path, sep=r"\s+|,|;|\t", engine="python", header=None, comment="#", on_bad_lines="error")
    if raw.empty:
        raise DatasetLoadError(f"Empty {kind} file: {path}")
    first = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
    known = set(names) | {"lat", "lon", "lng", "time", "timestamp"}
    has_header = any(x in known for x in first)
    if has_header:
        raw = pd.read_csv(path, sep=r"\s+|,|;|\t", engine="python", comment="#", on_bad_lines="error")
        raw.columns = [str(x).strip().lower() for x in raw.columns]
        aliases = {"lon": "longitude", "lng": "longitude", "lat": "latitude", "time": "timestamp"}
        raw = raw.rename(columns=aliases)
        missing = set(names) - set(raw.columns)
        if missing:
            raise DatasetLoadError(f"{kind} missing columns {sorted(missing)}: {path}")
        preserved_id = None
        if kind == "arcs" and "arc_id" in raw.columns:
            preserved_id = pd.to_numeric(raw["arc_id"], errors="coerce")
        if kind == "nodes" and "node_id" in raw.columns:
            preserved_id = pd.to_numeric(raw["node_id"], errors="coerce")
        result = raw[names].copy()
    else:
        if raw.shape[1] != len(names):
            raise DatasetLoadError(f"{kind} expected {len(names)} columns, found {raw.shape[1]}: {path}")
        raw.columns = names
        result = raw
    if kind == "arcs":
        result.insert(0, "arc_id", preserved_id if 'preserved_id' in locals() and preserved_id is not None else pd.RangeIndex(len(result), name="arc_id"))
    elif kind == "nodes":
        result.insert(0, "node_id", preserved_id if 'preserved_id' in locals() and preserved_id is not None else pd.RangeIndex(len(result), name="node_id"))
    return result.reset_index(drop=True)


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
