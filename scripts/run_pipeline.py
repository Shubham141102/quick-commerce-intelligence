"""Run pipeline stages.

    python -m scripts.run_pipeline --stages ingest                 # latest generation run -> Bronze
    python -m scripts.run_pipeline --stages ingest --force-reload  # reload files already loaded
    python -m scripts.run_pipeline --stages ingest --reset-bronze  # rebuild Bronze from scratch
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.orchestration.pipeline import STAGES, run_pipeline
from src.orchestration.tracking import read_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quick-commerce pipeline.")
    parser.add_argument("--stages", default="ingest", help=f"comma-separated, from {STAGES}")
    parser.add_argument("--generation-run", default=None, help="generation run id (default: data/generation/LATEST)")
    parser.add_argument("--force-reload", action="store_true", help="re-ingest files already loaded")
    parser.add_argument("--reset-bronze", action="store_true", help="delete Bronze and its file-load log first")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_pipeline([s.strip() for s in args.stages.split(",")], args.generation_run,
                          force_reload=args.force_reload, reset_bronze=args.reset_bronze)
    print(f"run {result.run_id}: {result.status} in {time.perf_counter() - t0:.1f}s "
          f"(generation run {result.generation_dir.name})")
    for stage, summary in result.results.items():
        print(f"  {stage}: {summary}")

    from src.common.config import load_config
    from src.common.paths import resolve
    tables = read_meta(resolve(load_config("small").paths["metadata"]), "meta_table_runs")
    tables = tables[tables["run_id"] == result.run_id]
    if not tables.empty:
        summary = (tables.assign(rows_written=tables["rows_written"].astype(int),
                                 rows_corrupt=tables["rows_corrupt"].replace("", "0").astype(int))
                   .groupby("target")[["rows_written", "rows_corrupt"]].sum())
        with pd.option_context("display.max_rows", 50):
            print(summary.to_string())


if __name__ == "__main__":
    main()
