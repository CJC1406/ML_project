import argparse
from pathlib import Path
from .data.schema import TripFiles, validate_trip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    files = TripFiles(root / "trip.track", root / "trip.nodes", root / "trip.arcs", root / "trip.route")
    validate_trip(files)
    print("Dataset contract valid; no model training performed.")


if __name__ == "__main__":
    main()
