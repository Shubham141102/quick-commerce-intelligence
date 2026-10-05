"""Lost-sales estimate (Phase 5A): how much a focus SKU would have sold on its stockout days.

expected_units = average daily units sold on the SKU's in-stock days in the previous 28 days
                 (stockout days are excluded: their sales were cut short by empty shelves)
lost_units     = max(0, expected_units − units actually sold), on stockout days only
lost_revenue   = lost_units × catalog price

This is revenue lost for that SKU. In the simulation a shopper facing an empty shelf often bought a
substitute, so the store as a whole lost less than this figure.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import read_csv_strings
from src.ml.tables import LOST_SALES, read_table, write_table
from src.orchestration.tracking import RunTracker, utc_now
from src.transformations.gold.inventory import INVENTORY_DAILY

LOOKBACK_DAYS = 28
MIN_IN_STOCK_DAYS = 7
PAIR = ["store_id", "product_id"]


def estimate_lost_sales(inventory: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.sort_values([*PAIR, "business_date"]).copy()
    in_stock_units = inv["units_sold"].where(~inv["is_stockout"].astype(bool))
    inv["expected_units"] = (in_stock_units.groupby([inv["store_id"], inv["product_id"]])
                             .transform(lambda s: s.shift(1).rolling(LOOKBACK_DAYS, min_periods=MIN_IN_STOCK_DAYS).mean()))
    out = inv[inv["is_stockout"].astype(bool) & inv["expected_units"].notna()].copy()
    out["lost_units"] = np.maximum(0.0, out["expected_units"] - out["units_sold"]).round(3)
    out["expected_units"] = out["expected_units"].round(3)
    prices = products[["product_id", "category_id", "price"]].copy()
    prices["unit_price"] = prices["price"].astype(float)
    out = out.merge(prices[["product_id", "category_id", "unit_price"]], on="product_id")
    out["lost_revenue"] = (out["lost_units"] * out["unit_price"]).round(2)
    return out[[*PAIR, "business_date", "category_id", "units_sold", "expected_units", "lost_units", "unit_price",
                "lost_revenue"]]


def run_lost_sales(tracker: RunTracker, gold_root: Path, silver_root: Path) -> dict:
    t0, started = time.perf_counter(), utc_now()
    inventory = read_table(gold_root, INVENTORY_DAILY)
    products = pd.concat([read_csv_strings(p) for p in (silver_root / "slv_products").glob("*.csv")])
    lost = estimate_lost_sales(inventory, products)
    lost["model_version"] = f"lost_sales_{tracker.run_id}"
    n = write_table(lost, gold_root, LOST_SALES)
    tracker.table(stage="ml", job=LOST_SALES.name, source=",".join(LOST_SALES.sources), target=LOST_SALES.name,
                  engine="python", module="src.ml.lost_sales", rows_written=n, status="success", started_at=started,
                  ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
    for source in LOST_SALES.sources:
        tracker.lineage(source, LOST_SALES.name, LOST_SALES.name, engine="python")
    return {"stockout_days": n, "lost_units": round(float(lost["lost_units"].sum()), 1),
            "lost_revenue": round(float(lost["lost_revenue"].sum()), 2)}
