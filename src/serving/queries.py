"""The only data functions app pages may call (Project_Plan_v2.md §9.5, §10.1).

Each function takes the caller's role first and checks it (`require`) before querying, so access control
does not depend on the page menu. Queries run in DuckDB over the published snapshot with bound parameters.
"""

from __future__ import annotations

from functools import wraps

import pandas as pd

from src.serving.db import get_snapshot
from src.serving.permissions import require

TIER_ORDER = "CASE r.risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END"


def workspace(name: str):
    def decorate(fn):
        @wraps(fn)
        def wrapper(role: str | None, *args, **kwargs):
            require(role, name)
            return fn(*args, **kwargs)
        return wrapper
    return decorate


def _q(sql: str, params: list | None = None) -> pd.DataFrame:
    return get_snapshot().query(sql, params)


def snapshot_info() -> dict[str, str]:
    return dict(get_snapshot().manifest)


# ----------------------------------------------------------------------------------- filters

@workspace("inventory")
def stores() -> pd.DataFrame:
    return _q("SELECT store_id, store_name, city FROM dim_stores ORDER BY store_id")


@workspace("inventory")
def categories() -> pd.DataFrame:
    return _q("SELECT category_id, category_name FROM dim_categories ORDER BY category_name")


@workspace("inventory")
def focus_products(store_id: str) -> pd.DataFrame:
    return _q("""SELECT DISTINCT r.product_id, p.product_name FROM gld_stockout_risk r
                 JOIN dim_products p USING (product_id) WHERE r.store_id = ? ORDER BY p.product_name""", [store_id])


# ----------------------------------------------------------------------------------- overview

@workspace("inventory")
def inventory_kpis() -> dict:
    risk = _q("""SELECT max(as_of_date) AS as_of, count(*) AS skus,
                        count(*) FILTER (WHERE closing_stock = 0) AS zero_stock,
                        count(*) FILTER (WHERE closing_stock <= reorder_level) AS below_reorder,
                        count(*) FILTER (WHERE risk_tier = 'High') AS high,
                        count(*) FILTER (WHERE risk_tier = 'Medium') AS medium
                 FROM gld_stockout_risk WHERE run_type = 'current'""").iloc[0]
    orders = _q("""SELECT count(*) FILTER (WHERE suggested_qty > 0) AS skus_to_order, coalesce(sum(suggested_qty), 0) AS units
                   FROM gld_replenishment WHERE run_type = 'current'""").iloc[0]
    forecast = _q("""SELECT min(forecast_date) AS first_day, max(forecast_date) AS last_day, sum(forecast_units) AS units
                     FROM gld_demand_predictions WHERE run_type = 'future'""").iloc[0]
    return {**risk.to_dict(), "skus_to_order": int(orders["skus_to_order"]), "units_to_order": int(orders["units"]),
            "forecast_first_day": forecast["first_day"], "forecast_last_day": forecast["last_day"],
            "forecast_units_7d": float(forecast["units"])}


@workspace("inventory")
def stock_health_by_store() -> pd.DataFrame:
    return _q("""SELECT s.store_name AS store, s.city,
                        count(*) FILTER (WHERE risk_tier = 'High') AS high,
                        count(*) FILTER (WHERE risk_tier = 'Medium') AS medium,
                        count(*) FILTER (WHERE risk_tier = 'Low') AS low,
                        count(*) FILTER (WHERE closing_stock = 0) AS zero_stock,
                        round(median(days_of_cover), 1) AS median_days_of_cover
                 FROM gld_stockout_risk r JOIN dim_stores s USING (store_id)
                 WHERE run_type = 'current' GROUP BY ALL ORDER BY high DESC, medium DESC, store""")


# ----------------------------------------------------------------------------------- forecasting

@workspace("inventory")
def category_forecast(store_id: str, category_id: str) -> pd.DataFrame:
    """Daily actual units, the September 1-day-ahead backtest forecast, and the future forecast with its band."""
    return _q("""SELECT s.business_date AS day, s.units AS actual, b.forecast_units AS backtest_forecast,
                        NULL::DOUBLE AS future_forecast, NULL::DOUBLE AS lower, NULL::DOUBLE AS upper
                 FROM gld_daily_category_sales s
                 LEFT JOIN gld_demand_predictions b ON b.store_id = s.store_id AND b.category_id = s.category_id
                      AND b.forecast_date = s.business_date AND b.run_type = 'backtest' AND b.horizon = 1
                 WHERE s.store_id = ? AND s.category_id = ?
                 UNION ALL
                 SELECT forecast_date, NULL, NULL, forecast_units, lower_units, upper_units FROM gld_demand_predictions
                 WHERE run_type = 'future' AND store_id = ? AND category_id = ?
                 ORDER BY day""", [store_id, category_id, store_id, category_id])


@workspace("inventory")
def forecast_accuracy() -> pd.DataFrame:
    return _q("""SELECT evaluation, model, horizon, wape, mae, bias, interval_coverage, rows
                 FROM gld_forecast_metrics WHERE segment = 'all' ORDER BY evaluation, horizon, model""")


@workspace("inventory")
def accuracy_by_category() -> pd.DataFrame:
    return _q("""SELECT c.category_name AS category,
                        max(wape) FILTER (WHERE model = 'model') AS model_wape,
                        max(wape) FILTER (WHERE model = 'baseline_moving_avg') AS moving_avg_wape,
                        max(wape) FILTER (WHERE model = 'baseline_seasonal_naive') AS seasonal_naive_wape
                 FROM gld_forecast_metrics m JOIN dim_categories c ON c.category_id = m.segment
                 WHERE evaluation = 'category_daily' AND horizon = 0 GROUP BY ALL ORDER BY model_wape""")


@workspace("inventory")
def feature_importance() -> pd.DataFrame:
    return _q("SELECT feature, rank, importance_mae, importance_std FROM gld_forecast_feature_importance ORDER BY rank")


# ----------------------------------------------------------------------------------- stockout & replenishment

@workspace("inventory")
def risk_list(store_ids: list[str] | None = None, tiers: list[str] | None = None) -> pd.DataFrame:
    sql = """SELECT r.risk_tier, s.store_name AS store, p.product_name AS product, c.category_name AS category,
                     r.closing_stock, r.reorder_level, round(r.forecast_demand_3d, 1) AS forecast_3d,
                     r.days_of_cover, rp.suggested_qty, r.reason_codes, r.store_id, r.product_id
              FROM gld_stockout_risk r
              JOIN gld_replenishment rp USING (as_of_date, store_id, product_id)
              JOIN dim_stores s USING (store_id) JOIN dim_products p USING (product_id)
              JOIN dim_categories c ON c.category_id = r.category_id
              WHERE r.run_type = 'current'"""
    params: list = []
    if store_ids:
        sql += f" AND r.store_id IN ({', '.join('?' * len(store_ids))})"
        params += store_ids
    if tiers:
        sql += f" AND r.risk_tier IN ({', '.join('?' * len(tiers))})"
        params += tiers
    return _q(sql + f" ORDER BY {TIER_ORDER}, r.days_of_cover NULLS FIRST, store, product", params)


@workspace("inventory")
def sku_timeline(store_id: str, product_id: str) -> pd.DataFrame:
    """Daily closing stock, units sold and reorder level, followed by the 7-day demand forecast."""
    return _q("""SELECT business_date AS day, closing_stock, units_sold, reorder_level,
                        NULL::DOUBLE AS forecast, NULL::DOUBLE AS lower, NULL::DOUBLE AS upper
                 FROM gld_inventory_daily WHERE store_id = ? AND product_id = ?
                 UNION ALL
                 SELECT forecast_date, NULL, NULL, NULL, forecast_units, lower_units, upper_units
                 FROM gld_sku_demand_forecast WHERE run_type = 'future' AND store_id = ? AND product_id = ?
                 ORDER BY day""", [store_id, product_id, store_id, product_id])


@workspace("inventory")
def replenishment_assumptions() -> dict:
    row = _q("""SELECT any_value(lead_time_days) AS lead_time_days, any_value(review_days) AS review_days,
                       any_value(service_level) AS service_level FROM gld_replenishment""").iloc[0]
    return row.to_dict()


@workspace("inventory")
def explorer_timeline(store_id: str, product_id: str) -> pd.DataFrame:
    """Every day of one store × SKU: stock movements, weekly count vs calculated stock, stockout, lost sales."""
    return _q("""SELECT i.business_date AS day, i.opening_stock, i.units_sold, i.restocked, i.damaged, i.adjusted,
                        i.closing_stock, i.reorder_level, i.is_stockout, i.snapshot_stock, i.reconciliation_gap,
                        i.days_of_inventory, coalesce(l.lost_units, 0) AS lost_units,
                        coalesce(l.lost_revenue, 0) AS lost_revenue
                 FROM gld_inventory_daily i
                 LEFT JOIN gld_lost_sales l USING (store_id, product_id, business_date)
                 WHERE i.store_id = ? AND i.product_id = ? ORDER BY day""", [store_id, product_id])


@workspace("inventory")
def explorer_summary(store_id: str, product_id: str) -> dict:
    row = _q("""SELECT p.product_name, p.price, c.category_name,
                       count(*) FILTER (WHERE i.is_stockout) AS stockout_days,
                       sum(i.restocked) AS restocked, sum(i.damaged) AS damaged, sum(i.units_sold) AS units_sold,
                       round(avg(abs(i.reconciliation_gap)), 2) AS mean_abs_gap,
                       count(i.reconciliation_gap) AS counts_compared,
                       (SELECT coalesce(sum(lost_units), 0) FROM gld_lost_sales WHERE store_id = ? AND product_id = ?) AS lost_units,
                       (SELECT coalesce(sum(lost_revenue), 0) FROM gld_lost_sales WHERE store_id = ? AND product_id = ?) AS lost_revenue
                FROM gld_inventory_daily i JOIN dim_products p USING (product_id)
                JOIN dim_categories c ON c.category_id = p.category_id
                WHERE i.store_id = ? AND i.product_id = ? GROUP BY ALL""",
             [store_id, product_id, store_id, product_id, store_id, product_id]).iloc[0]
    return row.to_dict()


@workspace("inventory")
def lost_sales_by_store() -> pd.DataFrame:
    return _q("""SELECT s.store_name AS store, s.city, count(*) AS stockout_days,
                        round(sum(l.lost_units), 1) AS lost_units, sum(l.lost_revenue) AS lost_revenue
                 FROM gld_lost_sales l JOIN dim_stores s USING (store_id)
                 GROUP BY ALL ORDER BY lost_revenue DESC""")


@workspace("inventory")
def top_lost_sales_skus(limit: int = 10) -> pd.DataFrame:
    return _q("""SELECT s.store_name AS store, p.product_name AS product, c.category_name AS category,
                        count(*) AS stockout_days, round(sum(l.lost_units), 1) AS lost_units,
                        sum(l.lost_revenue) AS lost_revenue, l.store_id, l.product_id
                 FROM gld_lost_sales l JOIN dim_stores s USING (store_id) JOIN dim_products p USING (product_id)
                 JOIN dim_categories c ON c.category_id = l.category_id
                 GROUP BY ALL ORDER BY lost_revenue DESC LIMIT ?""", [limit])


@workspace("inventory")
def stockout_backtest() -> pd.DataFrame:
    return _q("""SELECT rule, pair_days, flagged, stockouts_next_3d, true_positives, precision, recall, flag_rate
                 FROM gld_stockout_backtest ORDER BY precision DESC""")
