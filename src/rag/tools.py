"""Data tools for the business assistant (Phase 6C): the ONLY way the assistant gets numbers.

Each tool wraps an existing role-checked function in src/serving/queries.py, so access rules are inherited:
calling a tool outside the user's workspace raises AccessDenied, exactly like the pages. Every tool
- validates its inputs (Pydantic + checks against the published lookups and the data period);
- returns a small table, formatted headline numbers, the source table(s), the "as of" date and the scope;
- never returns customer-level data (no customer ids; segment and retention answers are aggregates).
No free-form SQL: the router may only pick a tool from TOOLS and fill its parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.rag.entities import Lookups
from src.serving import queries as q
from src.serving.permissions import allowed_workspaces
from src.serving.reliability import RELIABILITY_ACTION, reliability

MAX_ROWS = 20


class ToolInputError(ValueError):
    """A parameter is missing or invalid; the message is shown to the user."""


class ToolParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store_ids: list[str] | None = None
    tiers: list[Literal["High", "Medium", "Low"]] | None = None
    category_id: str | None = None
    product_id: str | None = None
    detectors: list[Literal["demand_spike", "payment_failure", "store_outage"]] | None = None
    start: date | None = None
    end: date | None = None
    limit: int = Field(10, ge=1, le=MAX_ROWS)


def validate_params(raw: dict, lk: Lookups) -> ToolParams:
    """Pydantic types + business checks: known ids, dates inside the data, start ≤ end."""
    p = ToolParams(**{k: v for k, v in raw.items() if v not in (None, [], "")})
    if p.store_ids and set(p.store_ids) - set(lk.stores["store_id"]):
        raise ToolInputError(f"Unknown store: {sorted(set(p.store_ids) - set(lk.stores['store_id']))}")
    if p.category_id and p.category_id not in set(lk.categories["category_id"]):
        raise ToolInputError(f"Unknown category: {p.category_id}")
    if p.product_id and p.product_id not in set(lk.products["product_id"]):
        raise ToolInputError(f"Unknown product: {p.product_id}")
    for d in (p.start, p.end):
        if d and not lk.first_day <= d <= lk.last_day:
            raise ToolInputError(f"The data covers {lk.first_day:%d %b %Y} – {lk.last_day:%d %b %Y}; {d:%d %b %Y} is outside it.")
    if p.start and p.end and p.start > p.end:
        raise ToolInputError("The period starts after it ends.")
    return p


@dataclass
class ToolResult:
    tool: str
    title: str
    headline: dict[str, str]
    table: pd.DataFrame
    source: str
    as_of: str
    scope: str
    note: str = ""
    total_rows: int = 0      # rows before the table was cut to `limit` (6G: a partial list must say so)


@dataclass(frozen=True)
class Tool:
    name: str
    workspace: str
    title: str
    description: str
    run: Callable[[str, ToolParams, Lookups], ToolResult] = field(repr=False)
    needs: tuple[str, ...] = ()      # parameters that must be present


# ----------------------------------------------------------------------------------------------- helpers
def _rupees(v) -> str:
    return f"₹{float(v or 0):,.0f}"


def _pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{float(v):.1%}"


def _no_pii(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if "customer" in c.lower() and c.lower().endswith("id")])


def _names(lk: Lookups, store_ids) -> list[str]:
    return lk.stores.loc[lk.stores["store_id"].isin(store_ids or []), "store_name"].tolist()


def _store_scope(lk: Lookups, p: ToolParams) -> str:
    if not p.store_ids:
        return "all stores"
    return ", ".join(n.replace("QC Dark Store", "").strip() for n in _names(lk, p.store_ids))


def _period(lk: Lookups, p: ToolParams) -> tuple[date, date]:
    return p.start or lk.first_day, p.end or lk.last_day


def _period_scope(lk: Lookups, p: ToolParams) -> str:
    s, e = _period(lk, p)
    return f"{s:%d %b} – {e:%d %b %Y}"


def _only_stores(df: pd.DataFrame, lk: Lookups, p: ToolParams, column: str = "store") -> pd.DataFrame:
    return df[df[column].isin(_names(lk, p.store_ids))] if p.store_ids else df


def _result(tool: str, title: str, headline: dict, table: pd.DataFrame, source: str, as_of, scope: str,
            note: str = "", total: int | None = None) -> ToolResult:
    """`total` = rows before the caller cut the table to the requested limit (defaults to the table's length)."""
    as_of_text = f"{pd.Timestamp(as_of):%d %b %Y}" if as_of is not None else ""
    shown = _no_pii(table).head(MAX_ROWS).reset_index(drop=True)
    return ToolResult(tool, title, headline, shown, source, as_of_text, scope, note,
                      total if total is not None else len(table))


# ----------------------------------------------------------------------------------------------- inventory
def _inventory_overview(role, p, lk):
    k = q.inventory_kpis(role)
    table = _only_stores(q.stock_health_by_store(role), lk, p)
    return _result("inventory_overview", "Stock position", {
        "High risk": str(int(k["high"])), "Medium risk": str(int(k["medium"])), "Zero stock": str(int(k["zero_stock"])),
        "SKUs to reorder": f"{k['skus_to_order']} ({k['units_to_order']} units)",
        "Forecast, next 7 days": f"{k['forecast_units_7d']:,.0f} units"},
        table, "gld_stockout_risk, gld_replenishment, gld_demand_predictions", k["as_of"], _store_scope(lk, p))


def _at_risk_skus(role, p, lk):
    tiers = p.tiers or ["High", "Medium"]
    risks = q.risk_list(role, p.store_ids, tiers)
    k = q.inventory_kpis(role)
    counts = risks["risk_tier"].value_counts()
    table = risks[["risk_tier", "store", "product", "category", "closing_stock", "reorder_level", "forecast_3d",
                   "days_of_cover", "suggested_qty", "reason_codes"]].head(p.limit)
    headline = {f"{t} risk": str(int(counts.get(t, 0))) for t in tiers}
    headline["Units to order"] = str(int(risks["suggested_qty"].fillna(0).sum()))
    return _result("at_risk_skus", "SKUs at risk of running out", headline, table,
                   "gld_stockout_risk, gld_replenishment", k["as_of"],
                   f"{_store_scope(lk, p)} · {' + '.join(tiers)} tier", total=len(risks))


def _suggested_orders(role, p, lk):
    risks = q.risk_list(role, p.store_ids, p.tiers)
    k = q.inventory_kpis(role)
    orders = risks[risks["suggested_qty"] > 0]
    table = orders[["store", "product", "risk_tier", "closing_stock", "forecast_3d", "suggested_qty"]].head(p.limit)
    return _result("suggested_orders", "Suggested orders (advisory)", {
        "SKUs to order": str(len(orders)), "Units": str(int(orders["suggested_qty"].sum()))},
        table, "gld_replenishment", k["as_of"], _store_scope(lk, p),
        "Advisory only: the store manager approves every order.", total=len(orders))


def _category_forecast(role, p, lk):
    stores = p.store_ids or lk.stores["store_id"].tolist()
    parts = [q.category_forecast(role, s, p.category_id) for s in stores]
    fut = pd.concat([d.dropna(subset=["future_forecast"]) for d in parts], ignore_index=True)
    if fut.empty:
        raise ToolInputError("No forecast is available for that selection.")
    daily = fut.groupby("day", as_index=False)[["future_forecast", "lower", "upper"]].sum()
    table = pd.DataFrame({"date": pd.to_datetime(daily["day"]).dt.strftime("%a %d %b"),
                          "forecast (units)": daily["future_forecast"].round(1),
                          "likely low": daily["lower"].round(1), "likely high": daily["upper"].round(1)})
    cat = lk.categories.set_index("category_id").loc[p.category_id, "category_name"]
    acc = q.accuracy_by_category(role)
    row = acc[acc["category"] == cat]
    level = reliability(float(row["model_wape"].iloc[0])) if len(row) else None
    headline = {"Next 7 days": f"{daily['future_forecast'].sum():,.0f} units",
                "Per day": f"~{daily['future_forecast'].mean():.1f} units"}
    if level:
        headline["Reliability"] = level
    return _result("category_forecast", f"7-day demand forecast — {cat}", headline, table,
                   "gld_demand_predictions, gld_forecast_metrics", lk.last_day, f"{_store_scope(lk, p)} · {cat}",
                   RELIABILITY_ACTION[level] if level else "")


def _forecast_reliability(role, p, lk):
    acc = q.accuracy_by_category(role)
    acc["reliability"] = acc["model_wape"].map(reliability)
    if p.category_id:
        cat = lk.categories.set_index("category_id").loc[p.category_id, "category_name"]
        acc = acc[acc["category"] == cat]
    counts = acc["reliability"].value_counts()
    return _result("forecast_reliability", "Forecast reliability by category",
                   {f"{lvl} reliability": f"{int(counts.get(lvl, 0))} categories" for lvl in ("High", "Medium", "Low")},
                   acc[["category", "reliability", "model_wape", "moving_avg_wape"]].round(3),
                   "gld_forecast_metrics", lk.last_day, "September backtest",
                   "High < 0.55 ≤ Medium ≤ 0.70 < Low (error ÷ units sold).")


def _lost_sales(role, p, lk):
    by_store = _only_stores(q.lost_sales_by_store(role), lk, p)
    top_all = _only_stores(q.top_lost_sales_skus(role, 50), lk, p).drop(columns=["store_id", "product_id"])
    top = top_all.head(p.limit)
    return _result("lost_sales", "Estimated lost sales from stockouts", {
        "Lost sales": _rupees(by_store["lost_revenue"].sum()), "Lost units": f"{by_store['lost_units'].sum():,.0f}",
        "Stockout days": f"{int(by_store['stockout_days'].sum()):,}"},
        top, "gld_lost_sales", lk.last_day, _store_scope(lk, p),
        "Estimate before substitutes; totals are reliable, small-SKU rankings are not.", total=len(top_all))


# ----------------------------------------------------------------------------------------------- business
def _sales_overview(role, p, lk):
    s, e = _period(lk, p)
    k = q.sales_kpis(role, s, e, p.store_ids)
    cur, prev = k["current"], k["previous"]

    def change(key):
        a, b = cur.get(key), prev.get(key)
        return "" if a is None or not b else f" ({float(a) / float(b) - 1:+.1%} vs previous {k['days']} days)"

    return _result("sales_overview", "Sales performance", {
        "Net revenue": _rupees(cur["net_revenue"]) + change("net_revenue"),
        "Completed orders": f"{int(cur['orders_completed'] or 0):,}" + change("orders_completed"),
        "Average order value": _rupees(cur["aov"]) + change("aov"),
        "Cancellation rate": _pct(cur["cancellation_rate"])},
        q.revenue_by_store(role, s, e, p.store_ids), "gld_daily_sales", e,
        f"{_store_scope(lk, p)} · {_period_scope(lk, p)}")


def _top_products(role, p, lk):
    s, e = _period(lk, p)
    top = q.top_products(role, s, e, p.limit)
    return _result("top_products", f"Top {p.limit} products by revenue",
                   {"Top product": str(top.iloc[0, 0]) if len(top) else "—"}, top, "gld_product_performance", e,
                   f"all stores · months overlapping {_period_scope(lk, p)}")


def _delivery_overview(role, p, lk):
    s, e = _period(lk, p)
    d = q.delivery_kpis(role, s, e, p.store_ids)
    return _result("delivery_overview", "Delivery performance", {
        "On-time rate": _pct(d["on_time_rate"]),
        "Avg delivery time": "—" if d["avg_minutes"] is None else f"{d['avg_minutes']:.1f} min",
        "Failed deliveries": f"{int(d['failed'] or 0):,}", "Cancelled in transit": f"{int(d['cancelled_in_transit'] or 0):,}"},
        q.delivery_by_store(role, s, e, p.store_ids), "gld_delivery_metrics", e,
        f"{_store_scope(lk, p)} · {_period_scope(lk, p)}", "On time = delivered within 15 minutes of pickup.")


def _cancellations(role, p, lk):
    s, e = _period(lk, p)
    c = q.cancellations_by_reason(role, s, e, p.store_ids)
    total = int(c["cancellations"].sum()) if "cancellations" in c else 0
    return _result("cancellations", "Cancellations by reason", {"Cancellations": f"{total:,}"}, c,
                   "gld_cancellation_metrics", e, f"{_store_scope(lk, p)} · {_period_scope(lk, p)}")


def _anomalies(role, p, lk):
    found = q.anomaly_list(role, p.detectors, p.store_ids)
    counts = found["detector"].value_counts()
    return _result("anomalies", "Detected anomalies", {
        "Anomalies": str(len(found)), "High severity": str(int((found["severity"] == "High").sum())),
        **{d.replace("_", " ").capitalize(): str(int(n)) for d, n in counts.items()}},
        found.drop(columns=["observed", "expected"]).head(p.limit), "gld_sales_anomalies", lk.last_day,
        _store_scope(lk, p), "A signal to investigate, not proof of a problem.", total=len(found))


# ----------------------------------------------------------------------------------------------- marketing
def _segments(role, p, lk):
    prof = q.segment_profiles(role)
    return _result("segments", "Customer segments", {"Segments": str(len(prof)),
                   "Largest": f"{prof.iloc[0]['segment_label']} ({prof.iloc[0]['share']:.0%})"},
                   prof[["segment_label", "customers", "share", "avg_spend", "avg_orders", "campaign_idea"]],
                   "gld_segment_profiles", lk.last_day, "customers with at least one completed order")


def _bought_with(role, p, lk):
    name = lk.products.set_index("product_id").loc[p.product_id, "product_name"]
    partners = q.product_partners(role, p.product_id)
    note = "" if len(partners) else "No strong 'bought together' rule for this product (confidence ≥ 10%, lift ≥ 2)."
    return _result("bought_with", f"Often bought with {name}",
                   {"Rules": str(len(partners)), **({"Strongest": f"{partners.iloc[0]['bought_with']} "
                                                                 f"({partners.iloc[0]['confidence']:.0%} of baskets)"}
                                                    if len(partners) else {})},
                   partners, "gld_basket_rules", lk.last_day, name, note)


def _recommendation_accuracy(role, p, lk):
    m = q.recommendation_metrics(role).set_index("method")
    return _result("recommendation_accuracy", "Recommendation accuracy (September backtest)", {
        f"{meth.capitalize()} Precision@10": _pct(m.loc[meth, "precision_at_10"])
        for meth in ("hybrid", "popularity", "repeat") if meth in m.index},
        m.reset_index(), "gld_recommendation_metrics", lk.last_day, "customers who bought in both periods",
        "Precision@10 = share of the 10 suggestions the customer actually bought.")


def _retention(role, p, lk):
    k = q.marketing_kpis(role)
    status = q.retention_status(role).groupby("status", sort=False, as_index=False)[["customers", "total_spend"]].sum()
    return _result("retention", "Customer retention status", {
        "Active (≤ 14 days)": f"{int(k['active']):,}", "At risk or lapsed": f"{int(k['at_risk_or_lapsed']):,}",
        "Repeat rate": _pct(k["repeat_rate"])}, status, "gld_customer_retention", lk.last_day,
        "all customers", "Descriptive only: no churn prediction.")


def _promotions(role, p, lk):
    pm = q.promotion_metrics(role).sort_values("uplift_pct", ascending=False)
    if p.category_id:
        cat = lk.categories.set_index("category_id").loc[p.category_id, "category_name"]
        pm = pm[pm["category"] == cat]
    return _result("promotions", "Promotion results", {
        "Promotions": str(len(pm)), "Discount cost": _rupees(pm["discount_cost"].sum()),
        "Median uplift": "—" if pm.empty else f"{pm['uplift_pct'].median():+.1f}%"},
        pm[["name", "category", "discount_pct", "days_active", "discount_cost", "uplift_pct"]].head(p.limit),
        "gld_promotion_metrics", lk.last_day, "all stores · sorted by uplift, highest first",
        "Before / after comparison, not a controlled test.", total=len(pm))


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("inventory_overview", "inventory", "Stock position", "risk tiers, zero stock, reorders, 7-day forecast",
         _inventory_overview),
    Tool("at_risk_skus", "inventory", "SKUs at risk", "SKUs likely to run out, by store and tier", _at_risk_skus),
    Tool("suggested_orders", "inventory", "Suggested orders", "what and how much to reorder", _suggested_orders),
    Tool("category_forecast", "inventory", "Category forecast", "next 7 days for a category", _category_forecast,
         needs=("category_id",)),
    Tool("forecast_reliability", "inventory", "Forecast reliability", "how far to trust the forecast per category",
         _forecast_reliability),
    Tool("lost_sales", "inventory", "Lost sales", "sales lost to stockouts", _lost_sales),
    Tool("sales_overview", "business", "Sales performance", "net revenue, orders, AOV vs previous period",
         _sales_overview),
    Tool("top_products", "business", "Top products", "best-selling products by revenue", _top_products),
    Tool("delivery_overview", "business", "Delivery performance", "on-time rate and delivery times",
         _delivery_overview),
    Tool("cancellations", "business", "Cancellations", "cancellations by reason", _cancellations),
    Tool("anomalies", "business", "Anomalies", "unusual demand, payment failures, quiet hours", _anomalies),
    Tool("segments", "marketing", "Customer segments", "who our customers are", _segments),
    Tool("bought_with", "marketing", "Bought together", "products bought with a product", _bought_with,
         needs=("product_id",)),
    Tool("recommendation_accuracy", "marketing", "Recommendation accuracy", "how good the recommendations are",
         _recommendation_accuracy),
    Tool("retention", "marketing", "Retention", "active, at-risk and lapsed customers", _retention),
    Tool("promotions", "marketing", "Promotions", "promotion uplift and cost", _promotions),
)}


def available_tools(role: str) -> list[Tool]:
    allowed = set(allowed_workspaces(role))
    return [t for t in TOOLS.values() if t.workspace in allowed]


def run_tool(name: str, role: str, raw_params: dict, lk: Lookups) -> ToolResult:
    """Validate, check required parameters, run (the wrapped query checks the role)."""
    tool = TOOLS[name]
    params = validate_params(raw_params, lk)
    missing = [n for n in tool.needs if getattr(params, n) in (None, [])]
    if missing:
        what = {"category_id": "a category (e.g. Dairy & Eggs)", "product_id": "a product name"}
        raise ToolInputError(f"Please name {' and '.join(what.get(m, m) for m in missing)}.")
    return tool.run(role, params, lk)
