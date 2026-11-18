from pathlib import Path
import pytest
from pathsnap.data.schema import DatasetError, TripFiles, validate_trip
from pathsnap.matching.viterbi import viterbi


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_schema_validation(tmp_path):
    write(tmp_path / "track", "time,latitude,longitude\n")
    write(tmp_path / "nodes", "node_id,latitude,longitude\n")
    write(tmp_path / "arcs", "arc_id,from_node,to_node\n")
    validate_trip(TripFiles(tmp_path / "track", tmp_path / "nodes", tmp_path / "arcs"))


def test_schema_reports_missing_column(tmp_path):
    write(tmp_path / "track", "time,latitude\n")
    write(tmp_path / "nodes", "node_id,latitude,longitude\n")
    write(tmp_path / "arcs", "arc_id,from_node,to_node\n")
    with pytest.raises(DatasetError, match="longitude"):
        validate_trip(TripFiles(tmp_path / "track", tmp_path / "nodes", tmp_path / "arcs"))


def test_viterbi_prefers_connected_sequence():
    emissions = [{"A": .9, "B": .1}, {"A": .2, "B": .8}, {"A": .1, "B": .9}]
    transitions = {("A", "A"): .9, ("A", "B"): .1, ("B", "A"): .1, ("B", "B"): .9}
    assert viterbi(emissions, transitions) == ["A", "B", "B"]


def test_viterbi_rejects_no_valid_path():
    with pytest.raises(ValueError, match="no valid Viterbi path"):
        viterbi([{"A": 1.0}, {"B": 1.0}], {})
