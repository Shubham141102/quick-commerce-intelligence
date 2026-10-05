"""Single entry point for reading and writing project tables (Project_Plan_v2.md §2.1).

Phase 1 needs only the pandas side: formatting typed frames into CSV strings
and writing them atomically. Spark readers/writers are added in Phase 2.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.schemas import TableSchema

TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%SZ"
DATE_FMT = "%Y-%m-%d"


def _format_column(series: pd.Series, dtype: str) -> list[str]:
    if dtype == "timestamp":
        values = pd.to_datetime(series)
        out = values.dt.strftime(TIMESTAMP_FMT)
    elif dtype == "date":
        values = pd.to_datetime(series)
        out = values.dt.strftime(DATE_FMT)
    elif dtype == "decimal":
        out = series.map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
    elif dtype == "double":
        out = series.map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}".rstrip("0").rstrip("."))
    elif dtype == "int":
        out = series.map(lambda v: "" if pd.isna(v) else str(int(v)))
    else:
        out = series.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return ["" if pd.isna(v) else v for v in out.tolist()]


def to_source_strings(df: pd.DataFrame, schema: TableSchema, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Format a typed frame into the CSV string representation of `schema`.

    Columns listed in `keep` (generator-internal helpers such as `_slice_ts`)
    are carried through unformatted.
    """
    data = {c.name: _format_column(df[c.name], c.dtype) for c in schema.columns}
    out = pd.DataFrame(data, dtype=object)
    for col in keep:
        out[col] = df[col].to_numpy()
    return out


def write_csv(rows: pd.DataFrame, path: Path, columns: list[str], raw_lines: dict[int, str] | None = None) -> int:
    """Write rows (already strings) to `path` atomically; returns rows written.

    `raw_lines` maps a positional row index to a pre-built line written verbatim
    instead of the row (used to emit deliberately malformed CSV rows).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    raw_lines = raw_lines or {}
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(columns)
        for i, record in enumerate(rows[columns].itertuples(index=False, name=None)):
            if i in raw_lines:
                fh.write(raw_lines[i] + "\n")
            else:
                writer.writerow(record)
    os.replace(tmp, path)
    return len(rows)


def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read a CSV with every column as string; empty fields stay empty strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)
