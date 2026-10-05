"""Gold on the small profile: core completion checks, a hand-computed store-day, typed reads, idempotency."""

from decimal import Decimal

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import types as T  # noqa: E402

from scripts.check_gold import evaluate  # noqa: E402
from src.common.io import read_csv_strings  # noqa: E402
from src.orchestration.pipeline import run_pipeline  # noqa: E402
from src.transformations.gold.run import GOLD_TABLES, read_gold  # noqa: E402


@pytest.fixture(scope="module")
def gold_env(silver_env, small_run, spark):
    result = run_pipeline(["gold"], small_run.run_id, silver_env["gen_root"], overrides=silver_env["overrides"],
                          spark=spark)
    return {**silver_env, "gold_result": result}


def _read(folder):
    return pd.concat([read_csv_strings(p) for p in folder.glob("*.csv")], ignore_index=True)


def test_core_checks_pass(gold_env):
    core, _, _ = evaluate(gold_env["paths"], "small")
    failed = [(name, detail) for name, ok, detail in core if not ok]
    assert not failed, failed


def test_one_store_day_matches_a_hand_computation(gold_env):
    orders = _read(gold_env["silver"] / "slv_orders")
    store, day = orders.groupby(["store_id", "business_date"]).size().idxmax()
    todays = orders[(orders["store_id"] == store) & (orders["business_date"] == day)]
    completed = set(todays.loc[todays["is_completed"] == "true", "order_id"])
    items = _read(gold_env["silver"] / "slv_order_items").merge(
        _read(gold_env["silver"] / "slv_products")[["product_id", "price"]], on="product_id")
    items = items[items["order_id"].isin(completed)]
    gmv = sum(Decimal(q) * Decimal(p) if f == "true" else Decimal(a)
              for q, p, a, f in zip(items["quantity"], items["price"], items["line_amount"], items["dq_unit_price_mismatch"]))
    row = _read(gold_env["gold"] / "gld_daily_sales").set_index(["store_id", "business_date"]).loc[(store, day)]
    assert int(row["orders_placed"]) == len(todays)
    assert int(row["orders_completed"]) == len(completed)
    assert int(row["orders_cancelled"]) == int((todays["status"] == "cancelled").sum())
    assert Decimal(row["gmv"]) == gmv
    assert int(row["units"]) == int(items["quantity"].astype(int).sum())


def test_every_table_reads_back_with_its_types(gold_env, spark):
    for table in GOLD_TABLES:
        df = read_gold(spark, gold_env["gold"], table.name)
        assert [f.name for f in df.schema.fields] == table.column_names
        assert df.count() == gold_env["gold_result"].results["gold"][table.name], table.name
    sales = read_gold(spark, gold_env["gold"], "gld_daily_sales")
    assert isinstance(sales.schema["gmv"].dataType, T.DecimalType)
    assert isinstance(sales.schema["business_date"].dataType, T.DateType)


def test_days_of_inventory_is_never_infinite(gold_env):
    inv = _read(gold_env["gold"] / "gld_inventory_daily")
    no_sales = inv["avg_daily_units_14d"].astype(float) == 0
    assert (inv.loc[no_sales, "days_of_inventory"] == "").all()
    assert not inv["days_of_inventory"].str.contains("inf", case=False).any()


def test_rebuild_is_idempotent(gold_env, small_run, spark):
    before = {t: _read(gold_env["gold"] / t) for t in ["gld_daily_sales", "gld_inventory_daily", "gld_basket_pairs"]}
    run_pipeline(["gold"], small_run.run_id, gold_env["gen_root"], overrides=gold_env["overrides"], spark=spark)
    for t, frame in before.items():
        pd.testing.assert_frame_equal(_read(gold_env["gold"] / t), frame, obj=t)
