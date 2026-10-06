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

SALES_ANOMALIES = GoldTable(
    "gld_sales_anomalies", ("anomaly_id",),
    "Unusual events found by three statistical detectors (store outage, demand spike, payment failure): what was "
    "observed vs what is normal for that store / hour / product over the previous 28 days, and how unlikely it is. "
    "A signal to investigate, not proof of a problem.",
    ("slv_orders", "slv_order_items", "slv_payments", "slv_application_logs"),
    (Col("anomaly_id", "string", "A0001…"),
     Col("detector", "string", "store_outage / demand_spike / payment_failure"),
     Col("store_id", "string", "store (empty for network-wide payment failures)"),
     Col("product_id", "string", "product (demand spikes only)"),
     Col("business_date", "date", "day the anomaly starts (IST)"),
     Col("start_ts", "timestamp", "start (UTC)"), Col("end_ts", "timestamp", "end (UTC)"),
     Col("observed", "double", "observed count over the anomaly (orders / failed payments)"),
     Col("expected", "double", "expected count from the previous 28 days"),
     Col("p_value", "double", "Poisson probability of a count at least this extreme"),
     Col("score", "double", "−log10(p_value), capped at 20 (higher = more unusual)"),
     Col("severity", "string", "High (score ≥ 8) / Medium"),
     Col("description", "string", "plain-language summary"),
     Col("model_version", "string", "pipeline run that produced it")),
    None, engine="python")

ANOMALY_SERIES = GoldTable(
    "gld_anomaly_series", ("anomaly_id", "ts"),
    "Context series for each anomaly (observed vs expected): hourly for the anomaly day (outages, payment failures), "
    "daily for ±14 days (demand spikes). Used by the anomaly chart.",
    ("gld_sales_anomalies",),
    (Col("anomaly_id", "string", "anomaly"), Col("ts", "timestamp", "hour or day (UTC)"),
     Col("observed", "double", "observed count"), Col("expected", "double", "expected count")),
    None, engine="python")

_MKT = ("gld_customer_360", "gld_customer_category")

CUSTOMER_SEGMENTS = GoldTable(
    "gld_customer_segments", ("customer_id",),
    "Segment of every customer with at least one completed order (K-means on behaviour and category mix).",
    _MKT, (Col("customer_id", "string", "customer"), Col("segment_id", "int", "segment number"),
           Col("segment_label", "string", "name generated from the segment's measured traits"),
           Col("model_version", "string", "model run")), None, engine="python")

SEGMENT_PROFILES = GoldTable(
    "gld_segment_profiles", ("segment_id",),
    "What each segment looks like (original-scale averages, top categories) and a suggested campaign.",
    _MKT, (Col("segment_id", "int", "segment number"), Col("segment_label", "string", "generated name"),
           Col("customers", "int", "customers in the segment"), Col("share", "double", "share of segmented customers"),
           Col("avg_spend", "double", "average total spend (₹)"), Col("avg_orders", "double", "average completed orders"),
           Col("avg_order_value", "double", "average order value (₹)"),
           Col("avg_items_per_order", "double", "average distinct products per order"),
           Col("avg_recency_days", "double", "average days since last completed order"),
           Col("night_order_share", "double", "share of orders 21:00–01:59 IST"),
           Col("weekend_order_share", "double", "share of orders on weekends"),
           Col("promo_order_share", "double", "share of completed orders with a promotion"),
           Col("top_categories", "string", "top 3 categories by share of spend"),
           Col("campaign_idea", "string", "suggested campaign, from the profile (an idea, not a tested result)"),
           Col("model_version", "string", "model run")), None, engine="python")

SEGMENTATION_SELECTION = GoldTable(
    "gld_segmentation_selection", ("k",),
    "How the number of segments was chosen: silhouette and inertia for k = 3…8; stability for the chosen k.",
    _MKT, (Col("k", "int", "number of segments tried"), Col("silhouette", "double", "silhouette score (higher = better separated)"),
           Col("inertia", "double", "within-segment sum of squares"), Col("chosen", "boolean", "k used"),
           Col("stability_ari", "double", "mean adjusted Rand index vs 5 other seeds (chosen k only)"),
           Col("model_version", "string", "model run")), None, engine="python")

BASKET_RULES = GoldTable(
    "gld_basket_rules", ("antecedent_id", "consequent_id"),
    "Association rules A → B from completed baskets (confidence ≥ 10%, lift ≥ 2): when A is bought, B often is too.",
    ("gld_basket_pairs",),
    (Col("antecedent_id", "string", "product A"), Col("antecedent", "string", "name of A"),
     Col("antecedent_category", "string", "category of A"), Col("consequent_id", "string", "product B"),
     Col("consequent", "string", "name of B"), Col("consequent_category", "string", "category of B"),
     Col("baskets_both", "int", "completed baskets with both"), Col("support", "double", "baskets_both ÷ all baskets"),
     Col("confidence", "double", "baskets_both ÷ baskets with A"), Col("lift", "double", "confidence ÷ support of B"),
     Col("model_version", "string", "pipeline run")), None, engine="python")

RECOMMENDATIONS = GoldTable(
    "gld_recommendations", ("customer_id", "rank"),
    "Top-10 product recommendations per customer (buy again + often bought with + popularity), with the reason.",
    ("slv_orders", "slv_order_items"),
    (Col("customer_id", "string", "customer"), Col("rank", "int", "1 = best"), Col("product_id", "string", "product"),
     Col("score", "double", "combined score"), Col("reason", "string", "you buy this often / often bought with your "
         "items / popular"), Col("model_version", "string", "model run")), None, engine="python")

RECOMMENDATION_METRICS = GoldTable(
    "gld_recommendation_metrics", ("method",),
    "September backtest of recommendations (trained on Apr–Aug; hybrid weights chosen on August): Precision@10, "
    "Recall@10, hit rate, coverage.",
    ("slv_orders", "slv_order_items"),
    (Col("method", "string", "hybrid (used) / repeat (buy again only) / popularity"),
     Col("weights", "string", "hybrid weights chosen on the August validation month"),
     Col("precision_at_10", "double", "share of the 10 recommendations bought in September"),
     Col("recall_at_10", "double", "share of September products that were recommended"),
     Col("hit_rate", "double", "customers with at least one recommended product bought"),
     Col("coverage", "double", "distinct products recommended ÷ catalog"),
     Col("customers", "int", "customers evaluated"), Col("model_version", "string", "model run")),
    None, engine="python")

RETENTION_COHORTS = GoldTable(
    "gld_retention_cohorts", ("cohort_month", "month_offset"),
    "Cohort retention: customers grouped by the month of their first completed order; share who ordered again "
    "in each later month. Descriptive (the synthetic data has no planted churn).",
    ("slv_orders",),
    (Col("cohort_month", "date", "month of first completed order"), Col("month_offset", "int", "months after the first"),
     Col("cohort_size", "int", "customers in the cohort"), Col("active_customers", "int", "with a completed order that month"),
     Col("retention_rate", "double", "active ÷ cohort size")), None, engine="python")

CUSTOMER_RETENTION = GoldTable(
    "gld_customer_retention", ("customer_id",),
    "Retention status of every customer at the end of the data: active / cooling / at risk / lapsed / never ordered, "
    "value tier, and whether their next order is overdue against their own rhythm.",
    ("slv_orders", "slv_customers"),
    (Col("customer_id", "string", "customer"), Col("first_order_date", "date", "first completed order"),
     Col("last_order_date", "date", "last completed order"), Col("orders_completed", "int", "completed orders"),
     Col("total_spend", "decimal", "order value of completed orders (₹)"),
     Col("days_since_last", "int", "days from last completed order to the end of the data"),
     Col("avg_days_between", "double", "average gap between completed orders"),
     Col("status", "string", "active (≤ 14 days) / cooling (15–30) / at_risk (31–60) / lapsed (> 60) / never_ordered"),
     Col("value_tier", "string", "High / Medium / Low by total spend (thirds)"),
     Col("overdue", "boolean", "days since last order > 2 × their average gap")), None, engine="python")

MARKETING_TABLES = (CUSTOMER_SEGMENTS, SEGMENT_PROFILES, SEGMENTATION_SELECTION, BASKET_RULES, RECOMMENDATIONS,
                    RECOMMENDATION_METRICS, RETENTION_COHORTS, CUSTOMER_RETENTION)

ML_TABLES: tuple[GoldTable, ...] = (DEMAND_PREDICTIONS, FORECAST_METRICS, FEATURE_IMPORTANCE, SKU_FORECAST,
                                    STOCKOUT_RISK, REPLENISHMENT, STOCKOUT_BACKTEST, LOST_SALES,
                                    SALES_ANOMALIES, ANOMALY_SERIES, *MARKETING_TABLES)


def read_table(gold_root: Path, table: GoldTable) -> pd.DataFrame:
    return read_typed(gold_root / table.name, table.typed_columns)


def write_table(frame: pd.DataFrame, gold_root: Path, table: GoldTable) -> int:
    schema = TableSchema(table.name, tuple(Column(c.name, c.dtype, True) for c in table.columns), table.grain)
    rows = to_source_strings(frame.sort_values(list(table.grain), kind="stable"), schema)
    return write_table_dir(rows, gold_root / table.name, table.column_names)
