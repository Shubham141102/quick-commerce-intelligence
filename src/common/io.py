"""Single entry point for reading and writing project tables (Project_Plan_v2.md §2.1).

Phase 1 needs only the pandas side: formatting typed frames into CSV strings
and writing them atomically. Spark readers/writers are added in Phase 2.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
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
    elif dtype == "boolean":
        out = series.map(lambda v: "" if pd.isna(v) else ("true" if bool(v) else "false"))
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
    replace_with_retry(tmp, path)
    return len(rows)


def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read a CSV with every column as string; empty fields stay empty strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def replace_with_retry(src: Path, dst: Path, attempts: int = 20, delay_s: float = 0.25) -> None:
    """os.replace that retries briefly: on Windows, antivirus scanning or file indexing can lock a file
    for a moment right after it is written, making the rename fail with 'Access is denied'."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def write_table_dir(rows: pd.DataFrame, final_dir: Path, columns: list[str]) -> int:
    """Write a whole table as `final_dir/part-00000.csv` via a staging folder + rename (pandas side of
    `spark_io.write_csv_atomic`). The previous version is kept in `<parent>/_previous/<name>`."""
    staging = final_dir.parent / "_staging" / final_dir.name
    if staging.exists():
        shutil.rmtree(staging)
    written = write_csv(rows, staging / "part-00000.csv", columns)
    if final_dir.exists():
        previous = final_dir.parent / "_previous" / final_dir.name
        if previous.exists():
            shutil.rmtree(previous)
        previous.parent.mkdir(parents=True, exist_ok=True)
        replace_with_retry(final_dir, previous)
    replace_with_retry(staging, final_dir)
    return written


def read_typed(path: Path, columns: list[tuple[str, str]]) -> pd.DataFrame:
    """Read one of our typed CSV tables (all part files under `path`) into pandas with proper dtypes."""
    parts = sorted(path.glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"no CSV files under {path}")
    frame = pd.concat([read_csv_strings(p) for p in parts], ignore_index=True)
    out = {}
    for name, dtype in columns:
        s = frame[name].replace("", np.nan)
        if dtype in ("int", "double", "decimal"):
            out[name] = pd.to_numeric(s)
        elif dtype == "boolean":
            out[name] = s.map({"true": True, "false": False})
        elif dtype in ("date", "timestamp"):
            out[name] = pd.to_datetime(s.str.replace("Z", "", regex=False))
        else:
            out[name] = frame[name]
    return pd.DataFrame(out)
