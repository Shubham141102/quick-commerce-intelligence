"""Gold tables written by the ML stage (Python / scikit-learn), declared like the Spark Gold tables so they
share the typed reader, the completion checks and the generated catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.io import read_typed, to_source_strings, write_table_dir
from src.common.schemas import Column, TableSchema
from src.transformations.gold.base import Col, GoldTable

_PRED_COMMON = (
    Col("forecast_date", "date", "day being forecast"),
    Col("origin_date", "date", "last day of known data when the forecast was made"),
    Col("horizon", "int", "days ahead (forecast_date − origin_date)"),
    Col("run_type", "string", "backtest (September, model trained on Apr–Aug) / future (after the data ends)"),
)

DEMAND_PREDICTIONS = GoldTable(
    "gld_demand_predictions", ("store_id", "category_id", "forecast_date", "origin_date"),
    "Demand forecasts per store × category × day, 1–7 days ahead: September backtest (made every day from "
    "1 to 30 Sep with a model trained on Apr–Aug, next to the actuals and both baselines) and the live 7-day "
    "forecast after the data ends.",
    ("gld_demand_features",),
    (Col("store_id", "string", "store"), Col("category_id", "string", "category"), *_PRED_COMMON,
     Col("forecast_units", "double", "model forecast (gradient boosting, Poisson loss)"),
     Col("lower_units", "double", "10th-percentile forecast (quantile model)"),
     Col("upper_units", "double", "90th-percentile forecast (quantile model)"),
     Col("baseline_seasonal_naive", "double", "same weekday one week earlier"),
     Col("baseline_moving_avg", "double", "mean of the 7 days up to origin_date"),
     Col("actual_units", "double", "units actually sold (empty for future forecasts)"),
     Col("model_version", "string", "model run that produced the row")),
    None, engine="python")

FORECAST_METRICS = GoldTable(
    "gld_forecast_metrics", ("evaluation", "model", "horizon", "segment"),
    "Backtest accuracy of the forecast and the baselines, overall and per category, by horizon.",
    ("gld_demand_predictions", "gld_sku_demand_forecast"),
    (Col("evaluation", "string", "category_daily (store × category) / sku_daily (store × focus SKU)"),
     Col("model", "string", "model / baseline_seasonal_naive / baseline_moving_avg"),
     Col("horizon", "int", "days ahead; 0 = all horizons 1–7 pooled"),
     Col("segment", "string", "all, or a category_id"),
     Col("mae", "double", "mean absolute error (units)"), Col("rmse", "double", "root mean squared error"),
     Col("wape", "double", "Σ|error| ÷ Σ actual (lower is better)"),
     Col("bias", "double", "Σ(forecast − actual) ÷ Σ actual (+ = over-forecast)"),
     Col("interval_coverage", "double", "share of actuals inside [lower, upper] (target ≈ 0.80); model only"),
     Col("rows", "int", "forecasts evaluated"), Col("model_version", "string", "model run")),
    None, engine="python")

FEATURE_IMPORTANCE = GoldTable(
    "gld_forecast_feature_importance", ("feature",),
    "Permutation importance of each forecasting feature on the September backtest (1 day ahead): how much "
    "the error grows when the feature is shuffled.",
    ("gld_demand_features",),
    (Col("feature", "string", "feature name"), Col("rank", "int", "1 = most important"),
     Col("importance_mae", "double", "increase in MAE when shuffled (units)"),
     Col("importance_std", "double", "standard deviation over repeats"),
     Col("model_version", "string", "model run")),
    None, engine="python")

SKU_FORECAST = GoldTable(
    "gld_sku_demand_forecast", ("store_id", "product_id", "forecast_date", "origin_date"),
    "Daily stock-depleting demand per store × focus SKU, 1–7 days ahead: the category forecast split by the "
    "SKU's share of the category over the 28 days before the origin (top-down).",
    ("gld_demand_predictions", "gld_inventory_daily", "gld_daily_category_sales", "slv_products"),
    (Col("store_id", "string", "store"), Col("product_id", "string", "focus SKU"),
     Col("category_id", "string", "category of the SKU"), *_PRED_COMMON,
     Col("sku_share", "double", "SKU units sold ÷ category units, 28 days before origin"),
     Col("forecast_units", "double", "category forecast × sku_share"),
     Col("lower_units", "double", "category lower bound × sku_share"),
     Col("upper_units", "double", "category upper bound × sku_share"),
     Col("baseline_moving_avg", "double", "SKU's mean daily units sold over the 7 days up to origin_date"),
     Col("actual_units", "double", "units actually sold (empty for future forecasts)"),
     Col("model_version", "string", "model run")),
    None, engine="python")

_AS_OF = (Col("as_of_date", "date", "decision date: end-of-day stock of this day is used"),
          Col("store_id", "string", "store"), Col("product_id", "string", "focus SKU"))

STOCKOUT_RISK = GoldTable(
    "gld_stockout_risk", ("as_of_date", "store_id", "product_id"),
    "Daily stockout-risk tier per store × focus SKU with reason codes (transparent rules on stock, reorder level, "
    "days of inventory and the SKU demand forecast). Advisory only.",
    ("gld_inventory_daily", "gld_sku_demand_forecast"),
    (*_AS_OF, Col("category_id", "string", "category of the SKU"),
     Col("closing_stock", "int", "end-of-day stock on as_of_date"),
     Col("reorder_level", "int", "reorder level"),
     Col("days_of_inventory", "double", "closing stock ÷ average daily units (last 14 days)"),
     Col("forecast_demand_3d", "double", "forecast units over the next 3 days (lead time 2 + 1)"),
     Col("forecast_upper_3d", "double", "90th-percentile forecast over the next 3 days"),
     Col("days_of_cover", "double", "closing stock ÷ forecast daily demand; empty when forecast is 0"),
     Col("risk_tier", "string", "High / Medium / Low"),
     Col("reason_codes", "string", "|-separated: ZERO_STOCK, DEMAND_EXCEEDS_STOCK, BELOW_REORDER, LOW_DAYS_COVER"),
     Col("run_type", "string", "backtest (September) / current (last day of data)"),
     Col("model_version", "string", "forecast model run used")),
    None, engine="python")

REPLENISHMENT = GoldTable(
    "gld_replenishment", ("as_of_date", "store_id", "product_id"),
    "Suggested replenishment quantity per store × focus SKU: forecast demand over lead time + review period + "
    "safety stock − current stock. Assumptions are stored on every row. Advisory only — no order is placed.",
    ("gld_inventory_daily", "gld_sku_demand_forecast"),
    (*_AS_OF, Col("closing_stock", "int", "end-of-day stock on as_of_date"),
     Col("forecast_demand_protection", "double", "forecast units over lead time + review period"),
     Col("demand_std_28d", "double", "standard deviation of daily units sold, last 28 days"),
     Col("safety_stock", "double", "z × demand_std_28d × √lead_time"),
     Col("suggested_qty", "int", "ceil(max(0, forecast_demand_protection + safety_stock − closing_stock))"),
     Col("risk_tier", "string", "tier from gld_stockout_risk"),
     Col("lead_time_days", "int", "assumed supplier lead time"), Col("review_days", "int", "assumed review period"),
     Col("service_level", "double", "target service level (z = 1.65 ≈ 95%)"),
     Col("run_type", "string", "backtest / current"), Col("model_version", "string", "forecast model run used")),
    None, engine="python")

STOCKOUT_BACKTEST = GoldTable(
    "gld_stockout_backtest", ("rule",),
    "September backtest of the risk rules on decisions taken while the SKU was still in stock: when a rule flags "
    "a SKU, does it actually stock out within the next 3 days? Compared with a naive reorder-level rule.",
    ("gld_stockout_risk", "gld_inventory_daily"),
    (Col("rule", "string", "risk rule evaluated"), Col("pair_days", "int", "store × SKU × day decisions evaluated"),
     Col("flagged", "int", "decisions flagged by the rule"),
     Col("stockouts_next_3d", "int", "decisions followed by a stockout within 3 days"),
     Col("true_positives", "int", "flagged and followed by a stockout"),
     Col("precision", "double", "true_positives ÷ flagged"), Col("recall", "double", "true_positives ÷ stockouts_next_3d"),
     Col("flag_rate", "double", "flagged ÷ pair_days"), Col("model_version", "string", "forecast model run used")),
    None, engine="python")

LOST_SALES = GoldTable(
    "gld_lost_sales", ("store_id", "product_id", "business_date"),
    "Estimated sales lost on stockout days per store × focus SKU: expected demand (average daily units on the "
    "SKU's in-stock days in the previous 28 days) minus units actually sold, valued at the catalog price. "
    "Only stockout days have rows. Revenue lost for that SKU, before any substitute the customer bought.",
    ("gld_inventory_daily", "slv_products"),
    (Col("store_id", "string", "store"), Col("product_id", "string", "focus SKU"),
     Col("business_date", "date", "stockout day (IST)"),
     Col("category_id", "string", "category of the SKU"),
     Col("units_sold", "int", "units actually sold that day"),
     Col("expected_units", "double", "average daily units on in-stock days, previous 28 days"),
     Col("lost_units", "double", "max(0, expected_units − units_sold)"),
     Col("unit_price", "decimal", "catalog price"),
     Col("lost_revenue", "decimal", "lost_units × unit_price"),
     Col("model_version", "string", "pipeline run that produced the estimate")),
    None, engine="python")

ML_TABLES: tuple[GoldTable, ...] = (DEMAND_PREDICTIONS, FORECAST_METRICS, FEATURE_IMPORTANCE, SKU_FORECAST,
                                    STOCKOUT_RISK, REPLENISHMENT, STOCKOUT_BACKTEST, LOST_SALES)


def read_table(gold_root: Path, table: GoldTable) -> pd.DataFrame:
    return read_typed(gold_root / table.name, table.typed_columns)


def write_table(frame: pd.DataFrame, gold_root: Path, table: GoldTable) -> int:
    schema = TableSchema(table.name, tuple(Column(c.name, c.dtype, True) for c in table.columns), table.grain)
    rows = to_source_strings(frame.sort_values(list(table.grain), kind="stable"), schema)
    return write_table_dir(rows, gold_root / table.name, table.column_names)
