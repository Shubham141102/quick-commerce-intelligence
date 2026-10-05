import pandas as pd

from src.common.io import read_csv_strings, write_csv


def test_csv_round_trip_keeps_commas_quotes_and_empty(tmp_path):
    rows = pd.DataFrame({
        "id": ["1", "2", "3"],
        "text": ['Fresh, well packed', 'Arrived in "10 minutes"', ""],
    })
    path = tmp_path / "t.csv"
    write_csv(rows, path, ["id", "text"])
    back = read_csv_strings(path)
    assert back.to_dict("records") == rows.to_dict("records")
    assert not (tmp_path / "t.csv.tmp").exists()


def test_raw_lines_are_written_verbatim(tmp_path):
    rows = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
    path = tmp_path / "t.csv"
    write_csv(rows, path, ["a", "b"], raw_lines={1: "broken,row,extra"})
    assert path.read_text().splitlines() == ["a,b", "x,1", "broken,row,extra"]
