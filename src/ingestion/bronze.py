"""The `ingest` stage: landing files -> Bronze tables (Project_Plan_v2.md §5).

Bronze is append-only and tied to one generation run. Ingesting a different
generation run on top would mix two unrelated synthetic worlds, so it is refused
unless `reset=True` (which clears Bronze, its checkpoints and the file-load log).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from src.ingestion.batch import ingest_batch
from src.ingestion.bronze_writer import BronzeContext
from src.ingestion.file_tracker import discover, loaded_identities
from src.ingestion.stream import ingest_stream
from src.orchestration.tracking import RunTracker


class BronzeSourceMismatch(RuntimeError):
    pass


def reset_bronze(bronze_root: Path, metadata_root: Path) -> None:
    shutil.rmtree(bronze_root, ignore_errors=True)
    shutil.rmtree(metadata_root / "meta_file_loads", ignore_errors=True)


def run_ingest(spark: SparkSession, tracker: RunTracker, generation_dir: Path, bronze_root: Path,
               metadata_root: Path, force_reload: bool = False) -> dict[str, int]:
    generation_run_id = generation_dir.name
    already_loaded, previous_runs = loaded_identities(metadata_root)
    other = previous_runs - {generation_run_id}
    if other:
        raise BronzeSourceMismatch(
            f"Bronze was built from {sorted(other)}; refusing to add {generation_run_id}. "
            "Re-run with --reset-bronze to rebuild Bronze from this generation run.")

    files = discover(generation_dir)
    ctx = BronzeContext(tracker, bronze_root, generation_run_id, {f.rel: f for f in files})
    rows_batch = ingest_batch(spark, ctx, files, already_loaded, force_reload)
    rows_stream = ingest_stream(spark, ctx, files, already_loaded, force_reload, generation_dir / "landing")
    shutil.rmtree(bronze_root / "_staging" / tracker.run_id, ignore_errors=True)
    return {"files_found": len(files), "rows_batch": rows_batch, "rows_stream": rows_stream}
