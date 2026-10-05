"""Read-only DuckDB access to the published snapshot (`data/demo/`).

The app never touches Spark or pipeline code: every CSV is loaded once into an in-memory DuckDB table,
typed from `snapshot_schema.csv`. Filters run inside DuckDB, so pages only receive the rows they show.
Set QCI_DEMO_DIR to point the app at another snapshot (used by tests).
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from src.common.paths import PROJECT_ROOT

DUCK_TYPES = {"string": "VARCHAR", "int": "BIGINT", "decimal": "DECIMAL(14,2)", "double": "DOUBLE",
              "date": "DATE", "timestamp": "TIMESTAMP", "boolean": "BOOLEAN"}


def demo_dir() -> Path:
    return Path(os.environ.get("QCI_DEMO_DIR", PROJECT_ROOT / "data" / "demo"))


class Snapshot:
    def __init__(self, folder: Path):
        if not (folder / "snapshot_schema.csv").exists():
            raise FileNotFoundError(f"no published snapshot in {folder}; run: python -m scripts.run_pipeline --stages publish")
        self.folder = folder
        self.con = duckdb.connect(database=":memory:")
        schema: dict[str, dict[str, str]] = {}
        with open(folder / "snapshot_schema.csv", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                schema.setdefault(row["table"], {})[row["column"]] = DUCK_TYPES[row["dtype"]]
        for table, columns in schema.items():
            path = (folder / f"{table}.csv").as_posix().replace("'", "''")
            cols = ", ".join(f"'{c}': '{t}'" for c, t in columns.items())
            # loaded into memory once (~27 MB): every later query is fast and never re-reads the files
            self.con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_csv('{path}', header = true, columns = {{{cols}}}, "
                f"timestampformat = '%Y-%m-%dT%H:%M:%SZ', dateformat = '%Y-%m-%d', nullstr = '')")
        self.tables = sorted(schema)
        with open(folder / "snapshot_manifest.csv", encoding="utf-8", newline="") as fh:
            self.manifest = {r["key"]: r["value"] for r in csv.DictReader(fh)}

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()


@lru_cache(maxsize=4)
def get_snapshot(folder: str | None = None) -> Snapshot:
    return Snapshot(Path(folder) if folder else demo_dir())
