"""Stockout risk and replenishment (Phase 4C): transparent rules on top of the SKU demand forecast.

Every day of the September backtest (and for the last day of data, "current"), each store × focus SKU gets:
- a risk tier with reason codes
    High   : stock is 0 (ZERO_STOCK), or forecast demand over lead time + 1 day ≥ stock (DEMAND_EXCEEDS_STOCK)
    Medium : stock ≤ reorder level (BELOW_REORDER), or days of inventory < 3 (LOW_DAYS_COVER)
    Low    : otherwise
- a suggested order: ceil(max(0, forecast over lead time + review period + safety stock − stock)),
  safety stock = z × std(daily units, last 28 days) × √lead time.
The backtest checks whether flagged SKUs really stocked out within 3 days, against a naive reorder-level rule.
Everything is advisory: nothing places an order.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.tables import (
    REPLENISHMENT,
    SKU_FORECAST,
    STOCKOUT_BACKTEST,
    STOCKOUT_RISK,
    read_table,
    write_table,
)
from src.orchestration.tracking import RunTracker, utc_now
from src.transformations.gold.inventory import INVENTORY_DAILY

LEAD_TIME_DAYS = 2
REVIEW_DAYS = 1
SERVICE_LEVEL, Z = 0.95, 1.65
LOW_COVER_DAYS = 3
OUTCOME_DAYS = 3        # backtest: did a stockout happen within this many days after the decision?
STD_WINDOW = 28
PAIR = ["store_id", "product_id"]


def _forecast_sums(sku: pd.DataFrame, days: int, col: str) -> pd.DataFrame:
    part = sku[sku["horizon"] <= days]
    out = part.groupby([*PAIR, "origin_date"], as_index=False)[col].sum()
    complete = part.groupby([*PAIR, "origin_date"])["horizon"].max().reset_index(name="__h")
    out = out.merge(complete, on=[*PAIR, "origin_date"])
    return out[out["__h"] == days].drop(columns="__h")


def build_risk(inventory: pd.DataFrame, sku: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = inventory.sort_values([*PAIR, "business_date"]).copy()
    inv["demand_std_28d"] = inv.groupby(PAIR)["units_sold"].transform(
        lambda s: s.rolling(STD_WINDOW, min_periods=7).std())
    d3 = _forecast_sums(sku, LEAD_TIME_DAYS + 1, "forecast_units").rename(columns={"forecast_units": "forecast_demand_3d"})
    u3 = _forecast_sums(sku, LEAD_TIME_DAYS + 1, "upper_units").rename(columns={"upper_units": "forecast_upper_3d"})
    dp = _forecast_sums(sku, LEAD_TIME_DAYS + REVIEW_DAYS, "forecast_units").rename(
        columns={"forecast_units": "forecast_demand_protection"})
    meta = sku.groupby([*PAIR, "origin_date"], as_index=False).agg(
        category_id=("category_id", "first"), run_type=("run_type", "first"), model_version=("model_version", "first"))
    decisions = (meta.merge(d3, on=[*PAIR, "origin_date"]).merge(u3, on=[*PAIR, "origin_date"])
                 .merge(dp, on=[*PAIR, "origin_date"])
                 .merge(inv[[*PAIR, "business_date", "closing_stock", "reorder_level", "days_of_inventory",
                             "demand_std_28d"]].rename(columns={"business_date": "origin_date"}), on=[*PAIR, "origin_date"])
                 .rename(columns={"origin_date": "as_of_date"}))
    decisions["run_type"] = decisions["run_type"].replace({"future": "current"})

    stock = decisions["closing_stock"]
    reasons = pd.DataFrame({
        "ZERO_STOCK": stock <= 0,
        "DEMAND_EXCEEDS_STOCK": decisions["forecast_demand_3d"] >= stock,
        "BELOW_REORDER": stock <= decisions["reorder_level"],
        "LOW_DAYS_COVER": decisions["days_of_inventory"] < LOW_COVER_DAYS,
    })
    decisions["reason_codes"] = reasons.apply(lambda r: "|".join(c for c in reasons.columns if r[c]), axis=1)
    high = reasons["ZERO_STOCK"] | reasons["DEMAND_EXCEEDS_STOCK"]
    medium = reasons["BELOW_REORDER"] | reasons["LOW_DAYS_COVER"]
    decisions["risk_tier"] = np.select([high, medium], ["High", "Medium"], default="Low")
    daily = decisions["forecast_demand_3d"] / (LEAD_TIME_DAYS + 1)
    decisions["days_of_cover"] = np.where(daily > 0, (stock / daily).round(2), np.nan)

    repl = decisions.copy()
    repl["safety_stock"] = (Z * repl["demand_std_28d"].fillna(0) * math.sqrt(LEAD_TIME_DAYS)).round(2)
    need = repl["forecast_demand_protection"] + repl["safety_stock"] - repl["closing_stock"]
    repl["suggested_qty"] = np.ceil(need.clip(lower=0).round(6)).astype(int)
    repl["lead_time_days"], repl["review_days"], repl["service_level"] = LEAD_TIME_DAYS, REVIEW_DAYS, SERVICE_LEVEL
    for col in ("forecast_demand_3d", "forecast_upper_3d", "forecast_demand_protection", "demand_std_28d"):
        decisions[col] = decisions[col].round(3)
        if col in repl:
            repl[col] = repl[col].round(3)
    return decisions, repl


def backtest(risk: pd.DataFrame, inventory: pd.DataFrame, version: str) -> pd.DataFrame:
    inv = inventory.sort_values([*PAIR, "business_date"])[[*PAIR, "business_date", "is_stockout"]].copy()
    # stockout in any of the next OUTCOME_DAYS days after the decision day
    inv["__s"] = inv["is_stockout"].astype(int)
    future = inv.groupby(PAIR)["__s"].transform(
        lambda s: s[::-1].rolling(OUTCOME_DAYS, min_periods=OUTCOME_DAYS).max()[::-1].shift(-1))
    inv["stockout_next"] = future
    bt = risk[risk["run_type"] == "backtest"].merge(
        inv[[*PAIR, "business_date", "stockout_next"]].rename(columns={"business_date": "as_of_date"}),
        on=[*PAIR, "as_of_date"]).dropna(subset=["stockout_next"])
    # Only decisions taken while the SKU is still in stock: warning *before* a stockout is the hard,
    # useful question (an item already at zero is trivially "at risk").
    bt = bt[bt["closing_stock"] > 0]
    outcome = bt["stockout_next"] == 1
    rules = {
        "High tier (forecast-based)": bt["risk_tier"] == "High",
        "High or Medium tier": bt["risk_tier"].isin(["High", "Medium"]),
        "Naive: stock <= reorder level": bt["closing_stock"] <= bt["reorder_level"],
    }
    rows = []
    for name, flag in rules.items():
        tp = int((flag & outcome).sum())
        rows.append({"rule": name, "pair_days": len(bt), "flagged": int(flag.sum()),
                     "stockouts_next_3d": int(outcome.sum()), "true_positives": tp,
                     "precision": round(tp / flag.sum(), 4) if flag.sum() else np.nan,
                     "recall": round(tp / outcome.sum(), 4) if outcome.sum() else np.nan,
                     "flag_rate": round(flag.mean(), 4), "model_version": version})
    return pd.DataFrame(rows)


def run_stockout(tracker: RunTracker, gold_root: Path) -> dict:
    t0, started = time.perf_counter(), utc_now()
    inventory = read_table(gold_root, INVENTORY_DAILY)
    sku = read_table(gold_root, SKU_FORECAST)
    version = sku["model_version"].iloc[0]
    risk, repl = build_risk(inventory, sku)
    bt = backtest(risk, inventory, version)
    n_risk = write_table(risk, gold_root, STOCKOUT_RISK)
    n_repl = write_table(repl, gold_root, REPLENISHMENT)
    write_table(bt, gold_root, STOCKOUT_BACKTEST)

    tracker.model_run(model_name="stockout_risk", model_version=f"stockout_rules_{tracker.run_id}",
                      method="transparent rules on stock, reorder level, days of inventory and SKU forecast",
                      trained_at=utc_now(), input_table="gld_inventory_daily|gld_sku_demand_forecast",
                      params=json.dumps({"lead_time_days": LEAD_TIME_DAYS, "review_days": REVIEW_DAYS,
                                         "service_level": SERVICE_LEVEL, "z": Z, "low_cover_days": LOW_COVER_DAYS,
                                         "outcome_days": OUTCOME_DAYS}),
                      metrics=json.dumps(bt.set_index("rule")[["precision", "recall", "flag_rate"]].to_dict("index")),
                      status="current", limitations="Rule-based; lead time is an assumption (the simulator uses 1–2 "
                      "days); stock between weekly counts is rebuilt and drifts when upstream rows were quarantined.")
    for r in bt.itertuples():
        for metric in ("precision", "recall", "flag_rate"):
            tracker.model_metric(model_version=f"stockout_rules_{tracker.run_id}", evaluation="stockout_backtest",
                                 model=r.rule, horizon=OUTCOME_DAYS, segment="all", metric=metric,
                                 value=getattr(r, metric))
    for table, n in ((STOCKOUT_RISK, n_risk), (REPLENISHMENT, n_repl), (STOCKOUT_BACKTEST, len(bt))):
        tracker.table(stage="ml", job=table.name, source=",".join(table.sources), target=table.name, engine="python",
                      module="src.ml.stockout", rows_written=n, status="success", started_at=started,
                      ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        for source in table.sources:
            tracker.lineage(source, table.name, table.name, engine="python")
    current = risk[risk["run_type"] == "current"]["risk_tier"].value_counts().to_dict()
    return {"risk_rows": n_risk, "replenishment_rows": n_repl, "current_tiers": current,
            "backtest": bt.set_index("rule")[["precision", "recall"]].to_dict("index")}
