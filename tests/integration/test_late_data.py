"""A record that arrives late (in a later file, after Bronze and Silver were built) must be picked up
by the next run, flagged late in Bronze, and dated by its event time in Silver."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from src.common.io import write_csv  # noqa: E402
from src.common.spark_io import read_bronze  # noqa: E402
from src.orchestration.pipeline import run_pipeline  # noqa: E402
from src.transformations.silver.engine import read_silver  # noqa: E402

LATE_ORDER = {"order_id": "ORD8000001", "customer_id": "CUST00001", "store_id": "S01",
              "order_ts": "2025-09-10T06:30:00Z", "status": "delivered", "total_amount": "250.00"}


def test_late_record_is_ingested_and_dated_by_event_time(small_run, spark, tmp_path):
    # isolated copy of the generated data, so the shared test environment is untouched
    gen_root = tmp_path / "generation"
    shutil.copytree(Path(small_run.out_dir), gen_root / small_run.run_id)
    paths = {"generation": str(gen_root), "bronze": str(tmp_path / "bronze"), "silver": str(tmp_path / "silver"),
             "quarantine": str(tmp_path / "quarantine"), "metadata": str(tmp_path / "metadata")}
    overrides = {"paths": paths}
    run_pipeline(["ingest", "silver"], small_run.run_id, gen_root, overrides=overrides, spark=spark)
    assert read_silver(spark, tmp_path / "silver", "orders").where(F.col("order_id") == LATE_ORDER["order_id"]).count() == 0

    # the order (event on 10 Sep) shows up in a file for 25 Sep 23:00
    late_file = gen_root / small_run.run_id / "landing" / "batch" / "orders" / "orders__2025-09-25T23-00.csv"
    write_csv(pd.DataFrame([LATE_ORDER]), late_file, list(LATE_ORDER))
    second = run_pipeline(["ingest", "silver"], small_run.run_id, gen_root, overrides=overrides, spark=spark)

    assert second.results["ingest"]["rows_batch"] == 1          # only the new file was loaded
    bronze = read_bronze(spark, tmp_path / "bronze", "orders").where(F.col("order_id") == LATE_ORDER["order_id"])
    assert [r["_is_late"] for r in bronze.collect()] == ["true"]
    silver = read_silver(spark, tmp_path / "silver", "orders").where(F.col("order_id") == LATE_ORDER["order_id"])
    rows = silver.collect()
    assert len(rows) == 1
    assert str(rows[0]["business_date"]) == "2025-09-10"       # event date, not arrival date
