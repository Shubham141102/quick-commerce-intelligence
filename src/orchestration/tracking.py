"""Run tracking: execution metadata written as CSV (Project_Plan_v2.md §12).

One file per run per table under `data/metadata/<table>/run_<run_id>.csv`, so runs
never append to the same file. Unknown values stay empty; nothing is invented.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.io import TIMESTAMP_FMT, read_csv_strings, write_csv

META_TABLES = {
    "meta_pipeline_runs": ["run_id", "pipeline", "profile", "generation_run_id", "stages", "status",
                           "started_at", "ended_at", "duration_s", "environment", "git_commit", "error"],
    "meta_stage_runs": ["run_id", "stage", "status", "started_at", "ended_at", "duration_s", "error"],
    "meta_table_runs": ["run_id", "stage", "job", "source", "target", "engine", "module", "batch_id",
                        "rows_read", "rows_written", "rows_rejected", "rows_rejected_cascade", "rows_deduplicated",
                        "rows_corrupt", "rows_flagged", "status", "started_at", "ended_at", "duration_s", "error"],
    "meta_quality_results": ["run_id", "dataset", "check_type", "rule", "column", "rows_affected", "rows_checked",
                             "outcome"],
    "meta_file_loads": ["run_id", "batch_id", "generation_run_id", "dataset", "slice", "file", "size_bytes",
                        "sha256", "rows_read", "rows_corrupt", "status", "loaded_at", "message"],
    "meta_lineage_edges": ["run_id", "source", "target", "transform", "engine"],
    "meta_model_runs": ["run_id", "model_name", "model_version", "method", "trained_at", "train_start", "train_end",
                        "test_start", "test_end", "input_table", "train_rows", "features", "params", "metrics",
                        "artifact_path", "status", "limitations"],
    "meta_model_metrics": ["run_id", "model_version", "evaluation", "model", "horizon", "segment", "metric", "value"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FMT)


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4]


class RunTracker:
    def __init__(self, metadata_root: Path, pipeline: str, profile: str, generation_run_id: str,
                 spark_version: str = "", git_commit: str = ""):
        self.metadata_root = metadata_root
        self.run_id = new_run_id()
        self.rows: dict[str, list[dict]] = {name: [] for name in META_TABLES}
        self._t0 = time.perf_counter()
        self._pipeline = {
            "run_id": self.run_id, "pipeline": pipeline, "profile": profile,
            "generation_run_id": generation_run_id, "started_at": utc_now(), "git_commit": git_commit,
            "environment": f"python={sys.version.split()[0]};spark={spark_version};os={platform.platform()}",
        }
        self._stages: list[str] = []

    @contextmanager
    def stage(self, name: str):
        started, t0 = utc_now(), time.perf_counter()
        self._stages.append(name)
        row = {"run_id": self.run_id, "stage": name, "started_at": started}
        try:
            yield
            row["status"] = "success"
        except BaseException as exc:
            row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["ended_at"] = utc_now()
            row["duration_s"] = round(time.perf_counter() - t0, 2)
            self.rows["meta_stage_runs"].append(row)

    def table(self, **fields) -> None:
        self.rows["meta_table_runs"].append({"run_id": self.run_id, **fields})

    def file_load(self, **fields) -> None:
        self.rows["meta_file_loads"].append({"run_id": self.run_id, **fields})

    def model_run(self, **fields) -> None:
        self.rows["meta_model_runs"].append({"run_id": self.run_id, **fields})

    def model_metric(self, **fields) -> None:
        self.rows["meta_model_metrics"].append({"run_id": self.run_id, **fields})

    def quality(self, **fields) -> None:
        self.rows["meta_quality_results"].append({"run_id": self.run_id, **fields})

    def lineage(self, source: str, target: str, transform: str, engine: str = "spark") -> None:
        edge = {"run_id": self.run_id, "source": source, "target": target, "transform": transform, "engine": engine}
        if edge not in self.rows["meta_lineage_edges"]:
            self.rows["meta_lineage_edges"].append(edge)

    def finish(self, status: str, error: BaseException | None = None) -> None:
        row = {**self._pipeline, "stages": "|".join(self._stages), "status": status,
               "ended_at": utc_now(), "duration_s": round(time.perf_counter() - self._t0, 2)}
        if error is not None:
            row["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        self.rows["meta_pipeline_runs"].append(row)
        for name, columns in META_TABLES.items():
            if self.rows[name]:
                frame = pd.DataFrame(self.rows[name]).reindex(columns=columns)
                frame = frame.astype(object).where(frame.notna(), "").astype(str)
                write_csv(frame, self.metadata_root / name / f"{self.run_id}.csv", columns)


def read_meta(metadata_root: Path, table: str) -> pd.DataFrame:
    """All runs of one metadata table (empty frame with the right columns if none yet)."""
    files = sorted((metadata_root / table).glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=META_TABLES[table])
    return pd.concat([read_csv_strings(f) for f in files], ignore_index=True)
