"""The only data functions app pages may call (Project_Plan_v2.md §9.5, §10.1).

Each function takes the caller's role first and checks it (`require`) before querying, so access control
does not depend on the page menu. Queries run in DuckDB over the published snapshot with bound parameters.
"""

from __future__ import annotations

from functools import wraps

import pandas as pd

from src.serving.db import get_snapshot
from src.serving.permissions import AccessDenied, require

TIER_ORDER = "CASE r.risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END"


def workspace(*names: str):
    """The caller's role must be allowed at least one of the named workspaces."""
    def decorate(fn):
        @wraps(fn)
        def wrapper(role: str | None, *args, **kwargs):
            for i, name in enumerate(names):
                try:
                    require(role, name)
                    break
                except AccessDenied:
                    if i == len(names) - 1:
                        raise
            return fn(*args, **kwargs)
        wrapper.workspaces = names
        return wrapper
    return decorate


def _in(column: str, values: list | None) -> tuple[str, list]:
    """SQL fragment ` AND column IN (?, …)` with its parameters (empty when no filter)."""
    if not values:
        return "", []
    return f" AND {column} IN ({', '.join('?' * len(values))})", list(values)


def _q(sql: str, params: list | None = None) -> pd.DataFrame:
    return get_snapshot().query(sql, params)


def snapshot_info() -> dict[str, str]:
    return dict(get_snapshot().manifest)


# ----------------------------------------------------------------------------------- filters

@workspace("inventory", "business", "marketing")
def stores() -> pd.DataFrame:
    return _q("SELECT store_id, store_name, city FROM dim_stores ORDER BY store_id")


@workspace("inventory", "business", "marketing")
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


# =================================================================================== business workspace

@workspace("business")
def period_bounds() -> dict:
    row = _q("SELECT min(business_date) AS first_day, max(business_date) AS last_day FROM gld_daily_sales").iloc[0]
    return row.to_dict()


@workspace("business")
def sales_kpis(start, end, store_ids: list[str] | None = None) -> dict:
    """Shared-metric KPIs for a date range, plus the same KPIs for the equally long period just before it."""
    f, p = _in("store_id", store_ids)
    sql = f"""SELECT sum(gmv) AS gmv, sum(discount) AS discount, sum(refunds) AS refunds, sum(net_revenue) AS net_revenue,
                     sum(orders_placed) AS orders_placed, sum(orders_completed) AS orders_completed,
                     sum(orders_cancelled) AS orders_cancelled, sum(units) AS units
              FROM gld_daily_sales WHERE business_date BETWEEN ? AND ?{f}"""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    prev_end, prev_start = pd.Timestamp(start) - pd.Timedelta(days=1), pd.Timestamp(start) - pd.Timedelta(days=days)

    def calc(a, b) -> dict:
        r = _q(sql, [a, b, *p]).iloc[0].to_dict()
        done = r["orders_completed"] or 0
        r["aov"] = float(r["gmv"] - r["discount"]) / done if done else None
        r["cancellation_rate"] = float(r["orders_cancelled"]) / r["orders_placed"] if r["orders_placed"] else None
        return r

    return {"current": calc(start, end), "previous": calc(prev_start.date(), prev_end.date()), "days": days}


@workspace("business")
def sales_trend(start, end, store_ids: list[str] | None = None, grain: str = "day") -> pd.DataFrame:
    unit = {"day": "day", "week": "week"}[grain]
    f, p = _in("store_id", store_ids)
    return _q(f"""SELECT date_trunc('{unit}', business_date)::DATE AS period, sum(net_revenue) AS net_revenue,
                         sum(gmv) AS gmv, sum(orders_completed) AS orders_completed, sum(orders_placed) AS orders_placed,
                         sum(units) AS units
                  FROM gld_daily_sales WHERE business_date BETWEEN ? AND ?{f} GROUP BY 1 ORDER BY 1""", [start, end, *p])


@workspace("business")
def revenue_by_store(start, end, store_ids: list[str] | None = None) -> pd.DataFrame:
    f, p = _in("d.store_id", store_ids)
    return _q(f"""SELECT s.store_name AS store, s.city, sum(d.net_revenue) AS net_revenue,
                         sum(d.orders_completed) AS orders_completed,
                         round(sum(d.gmv - d.discount) / nullif(sum(d.orders_completed), 0), 2) AS aov,
                         round(sum(d.orders_cancelled) / nullif(sum(d.orders_placed), 0), 4) AS cancellation_rate
                  FROM gld_daily_sales d JOIN dim_stores s USING (store_id)
                  WHERE d.business_date BETWEEN ? AND ?{f} GROUP BY ALL ORDER BY net_revenue DESC""", [start, end, *p])


@workspace("business")
def category_contribution(start, end, store_ids: list[str] | None = None) -> pd.DataFrame:
    f, p = _in("d.store_id", store_ids)
    return _q(f"""SELECT c.category_name AS category, sum(d.revenue) AS revenue, sum(d.units) AS units,
                         round(sum(d.revenue) / sum(sum(d.revenue)) OVER (), 4) AS share
                  FROM gld_daily_category_sales d JOIN dim_categories c USING (category_id)
                  WHERE d.business_date BETWEEN ? AND ?{f} GROUP BY ALL ORDER BY revenue DESC""", [start, end, *p])


@workspace("business")
def top_products(start, end, limit: int = 15) -> pd.DataFrame:
    """Network-wide top products for the months overlapping the range (product performance is monthly)."""
    return _q("""SELECT p.product_name AS product, c.category_name AS category, sum(pp.units) AS units,
                        sum(pp.revenue) AS revenue, round(sum(pp.revenue) / nullif(sum(pp.units), 0), 2) AS avg_price
                 FROM gld_product_performance pp JOIN dim_products p USING (product_id)
                 JOIN dim_categories c ON c.category_id = pp.category_id
                 WHERE pp.month BETWEEN date_trunc('month', ?::DATE) AND ?
                 GROUP BY ALL ORDER BY revenue DESC LIMIT ?""", [start, end, limit])


@workspace("business")
def delivery_kpis(start, end, store_ids: list[str] | None = None) -> dict:
    f, p = _in("store_id", store_ids)
    r = _q(f"""SELECT sum(deliveries) AS deliveries, sum(delivered) AS delivered, sum(failed) AS failed,
                      sum(cancelled_in_transit) AS cancelled_in_transit,
                      sum(on_time_rate * delivered) / nullif(sum(delivered), 0) AS on_time_rate,
                      sum(avg_delivery_minutes * delivered) / nullif(sum(delivered), 0) AS avg_minutes,
                      median(p50_delivery_minutes) AS typical_p50, median(p90_delivery_minutes) AS typical_p90
               FROM gld_delivery_metrics WHERE business_date BETWEEN ? AND ?{f}""", [start, end, *p]).iloc[0]
    return r.to_dict()


@workspace("business")
def delivery_trend(start, end, store_ids: list[str] | None = None) -> pd.DataFrame:
    f, p = _in("store_id", store_ids)
    return _q(f"""SELECT business_date AS day, sum(delivered) AS delivered, sum(failed) AS failed,
                         sum(on_time_rate * delivered) / nullif(sum(delivered), 0) AS on_time_rate,
                         sum(avg_delivery_minutes * delivered) / nullif(sum(delivered), 0) AS avg_minutes
                  FROM gld_delivery_metrics WHERE business_date BETWEEN ? AND ?{f}
                  GROUP BY 1 ORDER BY 1""", [start, end, *p])


@workspace("business")
def delivery_by_store(start, end, store_ids: list[str] | None = None) -> pd.DataFrame:
    f, p = _in("d.store_id", store_ids)
    return _q(f"""SELECT s.store_name AS store, sum(d.delivered) AS delivered, sum(d.failed) AS failed,
                         round(sum(d.on_time_rate * d.delivered) / nullif(sum(d.delivered), 0), 4) AS on_time_rate,
                         round(sum(d.avg_delivery_minutes * d.delivered) / nullif(sum(d.delivered), 0), 1) AS avg_minutes,
                         round(avg(d.active_partners), 1) AS avg_active_riders
                  FROM gld_delivery_metrics d JOIN dim_stores s USING (store_id)
                  WHERE d.business_date BETWEEN ? AND ?{f} GROUP BY ALL ORDER BY on_time_rate""", [start, end, *p])


@workspace("business")
def cancellations_by_reason(start, end, store_ids: list[str] | None = None) -> pd.DataFrame:
    f, p = _in("store_id", store_ids)
    return _q(f"""SELECT stage, reason, sum(cancellations) AS cancellations
                  FROM gld_cancellation_metrics
                  WHERE week_start BETWEEN date_trunc('week', ?::DATE) AND ?{f}
                  GROUP BY ALL ORDER BY cancellations DESC""", [start, end, *p])


@workspace("business")
def anomaly_summary() -> pd.DataFrame:
    return _q("""SELECT detector, count(*) AS anomalies, count(*) FILTER (WHERE severity = 'High') AS high_severity
                 FROM gld_sales_anomalies GROUP BY ALL ORDER BY detector""")


@workspace("business")
def anomaly_list(detectors: list[str] | None = None, store_ids: list[str] | None = None) -> pd.DataFrame:
    f1, p1 = _in("a.detector", detectors)
    f2, p2 = _in("a.store_id", store_ids)
    return _q(f"""SELECT a.anomaly_id, a.severity, a.score, a.detector, a.business_date,
                         coalesce(s.store_name, 'all stores') AS store, p.product_name AS product, a.description,
                         a.observed, a.expected
                  FROM gld_sales_anomalies a LEFT JOIN dim_stores s USING (store_id)
                  LEFT JOIN dim_products p USING (product_id)
                  WHERE TRUE{f1}{f2} ORDER BY a.score DESC, a.business_date""", [*p1, *p2])


@workspace("business")
def anomaly_series(anomaly_id: str) -> pd.DataFrame:
    return _q("""SELECT s.ts AS time, s.observed, s.expected, a.detector, a.start_ts, a.end_ts
                 FROM gld_anomaly_series s JOIN gld_sales_anomalies a USING (anomaly_id)
                 WHERE s.anomaly_id = ? ORDER BY s.ts""", [anomaly_id])


# ------------------------------------------------------------------------------------------- marketing (5D/5E)
@workspace("marketing")
def marketing_kpis() -> dict:
    row = _q("""SELECT count(*) AS customers,
                       count(*) FILTER (WHERE status <> 'never_ordered') AS buyers,
                       count(*) FILTER (WHERE orders_completed >= 2) AS repeat_buyers,
                       count(*) FILTER (WHERE status = 'active') AS active,
                       count(*) FILTER (WHERE status IN ('at_risk', 'lapsed')) AS at_risk_or_lapsed
                FROM gld_customer_retention""").iloc[0].to_dict()
    row["segments"] = int(_q("SELECT count(*) AS n FROM gld_segment_profiles").iloc[0]["n"])
    row["repeat_rate"] = row["repeat_buyers"] / row["buyers"] if row["buyers"] else None
    return row


@workspace("marketing")
def segment_profiles() -> pd.DataFrame:
    return _q("SELECT * EXCLUDE (model_version) FROM gld_segment_profiles ORDER BY customers DESC")


@workspace("marketing")
def segmentation_selection() -> pd.DataFrame:
    return _q("SELECT k, silhouette, inertia, chosen, stability_ari FROM gld_segmentation_selection ORDER BY k")


@workspace("marketing")
def segment_customers(segment_id: int, limit: int = 200) -> pd.DataFrame:
    return _q("""SELECT c.customer_id, c.city_id, c.orders_completed, c.total_spend, c.avg_order_value,
                        c.recency_days, r.status, c.top_category
                 FROM gld_customer_segments s JOIN gld_customer_360 c USING (customer_id)
                 LEFT JOIN gld_customer_retention r USING (customer_id)
                 WHERE s.segment_id = ? ORDER BY c.total_spend DESC LIMIT ?""", [int(segment_id), int(limit)])


@workspace("marketing")
def basket_rules(category_ids: list[str] | None = None, min_lift: float = 2.0, limit: int = 200) -> pd.DataFrame:
    f, p = _in("r.antecedent_category", category_ids)
    return _q(f"""SELECT r.antecedent, r.consequent, ca.category_name AS antecedent_category,
                         cb.category_name AS consequent_category, r.baskets_both, r.confidence, r.lift
                  FROM gld_basket_rules r
                  LEFT JOIN dim_categories ca ON ca.category_id = r.antecedent_category
                  LEFT JOIN dim_categories cb ON cb.category_id = r.consequent_category
                  WHERE r.lift >= ?{f} ORDER BY r.lift DESC, r.baskets_both DESC LIMIT ?""",
              [float(min_lift), *p, int(limit)])


@workspace("marketing")
def rule_products() -> pd.DataFrame:
    return _q("""SELECT DISTINCT antecedent_id AS product_id, antecedent AS product_name
                 FROM gld_basket_rules ORDER BY product_name""")


@workspace("marketing")
def product_partners(product_id: str) -> pd.DataFrame:
    return _q("""SELECT consequent AS bought_with, consequent_category AS category_id, baskets_both, confidence, lift
                 FROM gld_basket_rules WHERE antecedent_id = ? ORDER BY confidence DESC""", [product_id])


@workspace("marketing")
def recommendation_metrics() -> pd.DataFrame:
    return _q("""SELECT method, weights, precision_at_10, recall_at_10, hit_rate, coverage, customers
                 FROM gld_recommendation_metrics
                 ORDER BY CASE method WHEN 'hybrid' THEN 1 WHEN 'repeat' THEN 2 ELSE 3 END""")


@workspace("marketing")
def recommendation_customers(segment_id: int | None = None, limit: int = 300) -> pd.DataFrame:
    where, params = ("WHERE s.segment_id = ?", [int(segment_id)]) if segment_id is not None else ("", [])
    return _q(f"""SELECT s.customer_id, s.segment_label, c.orders_completed, c.total_spend
                  FROM gld_customer_segments s JOIN gld_customer_360 c USING (customer_id)
                  {where} ORDER BY c.total_spend DESC LIMIT ?""", [*params, int(limit)])


@workspace("marketing")
def customer_recommendations(customer_id: str) -> pd.DataFrame:
    return _q("""SELECT r.rank, p.product_name AS product, c.category_name AS category, r.reason, r.score
                 FROM gld_recommendations r LEFT JOIN dim_products p USING (product_id)
                 LEFT JOIN dim_categories c ON c.category_id = p.category_id
                 WHERE r.customer_id = ? ORDER BY r.rank""", [customer_id])


@workspace("marketing")
def segment_top_recommendations(segment_id: int, limit: int = 10) -> pd.DataFrame:
    return _q("""SELECT p.product_name AS product, count(*) AS customers,
                        count(*) FILTER (WHERE r.reason = 'often bought with your items') AS as_cross_sell
                 FROM gld_recommendations r JOIN gld_customer_segments s USING (customer_id)
                 LEFT JOIN dim_products p USING (product_id)
                 WHERE s.segment_id = ? AND r.reason <> 'you buy this often'
                 GROUP BY ALL ORDER BY customers DESC LIMIT ?""", [int(segment_id), int(limit)])


@workspace("marketing")
def retention_cohorts() -> pd.DataFrame:
    return _q("""SELECT cohort_month, month_offset, cohort_size, active_customers, retention_rate
                 FROM gld_retention_cohorts ORDER BY cohort_month, month_offset""")


@workspace("marketing")
def retention_status() -> pd.DataFrame:
    return _q("""SELECT status, value_tier, count(*) AS customers, sum(total_spend) AS total_spend
                 FROM gld_customer_retention GROUP BY status, value_tier
                 ORDER BY CASE status WHEN 'active' THEN 1 WHEN 'cooling' THEN 2 WHEN 'at_risk' THEN 3
                                      WHEN 'lapsed' THEN 4 ELSE 5 END, value_tier""")


@workspace("marketing")
def retention_customers(statuses: list[str] | None = None, tiers: list[str] | None = None,
                        overdue_only: bool = False, limit: int = 300) -> pd.DataFrame:
    f1, p1 = _in("r.status", statuses)
    f2, p2 = _in("r.value_tier", tiers)
    f3 = " AND r.overdue" if overdue_only else ""
    return _q(f"""SELECT r.customer_id, r.status, r.value_tier, r.orders_completed, r.total_spend,
                         r.last_order_date, r.days_since_last, r.avg_days_between, r.overdue, s.segment_label
                  FROM gld_customer_retention r LEFT JOIN gld_customer_segments s USING (customer_id)
                  WHERE r.status <> 'never_ordered'{f1}{f2}{f3}
                  ORDER BY r.total_spend DESC LIMIT ?""", [*p1, *p2, int(limit)])


@workspace("marketing")
def promotion_metrics() -> pd.DataFrame:
    return _q("""SELECT m.name, c.category_name AS category, m.discount_pct, m.start_date, m.end_date, m.days_active,
                        m.orders_with_promotion, m.discount_cost, m.promo_daily_units, m.baseline_daily_units,
                        m.uplift_pct
                 FROM gld_promotion_metrics m LEFT JOIN dim_categories c USING (category_id)
                 ORDER BY m.start_date""")
