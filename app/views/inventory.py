"""Inventory & Supply Chain workspace (Phase 4D): overview, demand forecasting, stockout risk & replenishment.

Every number comes from src/serving/queries.py (role-checked) over the published snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    explorer_chart,
    forecast_chart,
    importance_chart,
    stock_chart,
)
from app.components.insights import (  # noqa: E402
    RELIABILITY_ACTION,
    accuracy_headline,
    forecast_summary,
    reliability,
    stock_summary,
    style_reliability,
    style_tiers,
)
from app.components.ui import (  # noqa: E402
    advisory_note,
    label,
    notes,
    page_header,
    require_workspace,
    tab_intro,
)
from src.serving import queries as q  # noqa: E402

WS = "inventory"
SECTIONS = [  # (function, sidebar title, icon) — each is a page in the sidebar
    ("overview", "Overview", ":material/space_dashboard:"),
    ("forecasting", "Demand forecasting", ":material/trending_up:"),
    ("stockout", "Stockout risk & replenishment", ":material/production_quantity_limits:"),
    ("explorer", "Inventory explorer", ":material/manage_search:"),
]
_TITLES = {key: title for key, title, _ in SECTIONS}


def _open(key: str) -> str:
    """Guard + header for one section page; returns the role."""
    role = require_workspace(WS)
    page_header(WS, _TITLES[key])
    return role

# ------------------------------------------------------------------------------------------- overview
def overview() -> None:
    role = _open("overview")
    tab_intro("Where do we stand today: which SKUs are at risk and what should be reordered?")
    k = q.inventory_kpis(role)
    st.caption(f"As of **{k['as_of']:%d %b %Y}** · {k['skus']} store × focus-SKU pairs")
    c = st.columns(5)
    c[0].metric("High risk", int(k["high"]), help="Stock is 0, or forecast demand over lead time + 1 day ≥ stock")
    c[1].metric("Medium risk", int(k["medium"]), help="Stock ≤ reorder level, or under 3 days of inventory")
    c[2].metric("Zero stock", int(k["zero_stock"]))
    c[3].metric("SKUs to reorder", k["skus_to_order"], help=f"{k['units_to_order']} units in total (advisory)")
    c[4].metric("Forecast, next 7 days", f"{k['forecast_units_7d']:,.0f} units",
                help=f"All stores and categories, {k['forecast_first_day']:%d %b}–{k['forecast_last_day']:%d %b}")
    st.subheader("Stock health by store")
    st.dataframe(style_tiers(q.stock_health_by_store(role), count_columns={"high": "High", "medium": "Medium",
                                                                           "low": "Low"}), hide_index=True, width="stretch",
                 column_config={"median_days_of_cover": st.column_config.NumberColumn("median days of cover",
                                help="closing stock ÷ forecast daily demand")})
    advisory_note()

# ------------------------------------------------------------------------------------------- forecasting
def forecasting() -> None:
    role = _open("forecasting")
    tab_intro("How many units will each category sell in the next 7 days, and how far can we trust it?")
    stores, cats = q.stores(role), q.categories(role)
    a, b = st.columns(2)
    store = a.selectbox("Store", stores["store_id"], format_func=dict(zip(stores["store_id"], stores["store_name"])).get)
    cat = b.selectbox("Category", cats["category_id"], format_func=dict(zip(cats["category_id"], cats["category_name"])).get)
    fc = q.category_forecast(role, store, cat)
    full = st.toggle("Show full history (April–September)", value=False)
    st.plotly_chart(forecast_chart(fc, days_back=None if full else 30), width="stretch")
    notes("September: line vs bars = how close the model gets (trained on Apr–Aug only)",
          "After today: forecast line · band = likely range (8 in 10 days)")
    acc = q.forecast_accuracy(role)
    cat_acc = acc[acc["evaluation"] == "category_daily"]
    pooled = cat_acc[cat_acc["horizon"] == 0].set_index("model")["wape"]
    by_h = (cat_acc[cat_acc["horizon"] > 0].pivot_table(index="horizon", columns="model", values="wape")
            .reindex(columns=["model", "baseline_moving_avg", "baseline_seasonal_naive"]))
    by_cat = q.accuracy_by_category(role)
    by_cat["reliability"] = by_cat["model_wape"].map(reliability)
    cat_names = dict(zip(cats["category_id"], cats["category_name"]))
    picked = by_cat[by_cat["category"] == cat_names.get(cat)]

    lines, table = forecast_summary(fc)
    if len(picked):
        level = picked["reliability"].iloc[0]
        badge = {"High": ":green-badge[High]", "Medium": ":orange-badge[Medium]", "Low": ":red-badge[Low]"}[level]
        lines.append(f"**Forecast reliability for {cat_names[cat]}:** {badge} {RELIABILITY_ACTION[level]}")
    with st.container(border=True):
        label("Key takeaways")
        st.markdown("\n".join(f"- {line}" for line in lines))
        if len(table):
            st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Can I trust the forecast?")
    st.markdown(accuracy_headline(pooled, by_h))
    groups = (by_cat.groupby("reliability", sort=False)["category"].apply(lambda c: " · ".join(c))
              .reindex(["High", "Medium", "Low"]).dropna().reset_index())
    groups["what to do"] = groups["reliability"].map(RELIABILITY_ACTION)
    st.dataframe(style_reliability(groups.rename(columns={"category": "categories"})), hide_index=True,
                 width="stretch", column_config={"categories": st.column_config.TextColumn(width="large")})
    notes("Reliability = September error: High < 55% · Medium ≤ 70% · Low > 70% of units sold",
          "Staples forecast well · occasional purchases don't")

    with st.expander("Technical details (for analysts)"):
        m = st.columns(3)
        m[0].metric("Model WAPE", f"{pooled['model']:.3f}")
        m[1].metric("Moving average", f"{pooled['baseline_moving_avg']:.3f}",
                    delta=f"{pooled['baseline_moving_avg'] - pooled['model']:+.3f} worse", delta_color="off")
        m[2].metric("Seasonal naive", f"{pooled['baseline_seasonal_naive']:.3f}",
                    delta=f"{pooled['baseline_seasonal_naive'] - pooled['model']:+.3f} worse", delta_color="off")
        wape_cols = {"model": st.column_config.NumberColumn("model", format="%.3f"),
                     "baseline_moving_avg": st.column_config.NumberColumn("moving average", format="%.3f"),
                     "baseline_seasonal_naive": st.column_config.NumberColumn("seasonal naive", format="%.3f"),
                     "model_wape": st.column_config.NumberColumn("model", format="%.3f"),
                     "moving_avg_wape": st.column_config.NumberColumn("moving average", format="%.3f"),
                     "seasonal_naive_wape": st.column_config.NumberColumn("seasonal naive", format="%.3f")}
        st.markdown("**WAPE by days ahead** (lower is better)")
        st.dataframe(by_h.reset_index(), hide_index=True, width="stretch", column_config=wape_cols)
        st.markdown("**WAPE by category** (all horizons)")
        st.dataframe(style_reliability(by_cat[["category", "reliability", "model_wape", "moving_avg_wape",
                                               "seasonal_naive_wape"]]),
                     hide_index=True, width="stretch", column_config=wape_cols)
        st.markdown("**What drives the forecast** (permutation importance)")
        st.plotly_chart(importance_chart(q.feature_importance(role)), width="stretch")
        notes("**WAPE** — total error ÷ units sold (lower is better)",
              "**Noise floor** — ≈ 0.58: ~3 units/day per store-category is mostly chance",
              "**Baselines** — moving average = last 7 days · seasonal naive = same weekday last week",
              "**Assumptions** — weather = last 7-day average · no promotions after the data ends",
              "**Bias** — slight under-forecast (≈ −7%) · holidays approximate, no holiday effect",
              "Details: docs/05_demand_forecasting.md")

# ------------------------------------------------------------------------------------------- stockout
def stockout() -> None:
    role = _open("stockout")
    tab_intro("Which SKUs will run out before the next delivery, and how much should we order?")
    advisory_note()
    stores = q.stores(role)
    names = dict(zip(stores["store_id"], stores["store_name"]))
    f1, f2 = st.columns([2, 1])
    picked_stores = f1.multiselect("Stores", stores["store_id"], format_func=names.get, placeholder="All stores")
    tiers = f2.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    risks = q.risk_list(role, picked_stores or None, tiers or None)
    st.markdown(f"**{len(risks)} SKUs** · by tier, then fewest days of cover")
    st.dataframe(style_tiers(risks.drop(columns=["store_id", "product_id"]), ["risk_tier"]), hide_index=True,
                 width="stretch",
                 column_config={
                     "risk_tier": "tier", "forecast_3d": st.column_config.NumberColumn("forecast 3 days"),
                     "days_of_cover": st.column_config.NumberColumn("days of cover", format="%.1f"),
                     "suggested_qty": st.column_config.NumberColumn("suggested order"),
                     "reason_codes": "reasons"})
    st.download_button("Download suggested orders (CSV)",
                       risks[risks["suggested_qty"] > 0].drop(columns=["store_id", "product_id"]).to_csv(index=False),
                       file_name="suggested_orders.csv", mime="text/csv", disabled=risks.empty)

    st.subheader("Look at one SKU")
    s1, s2 = st.columns(2)
    sku_store = s1.selectbox("Store ", stores["store_id"], format_func=names.get,
                             index=list(stores["store_id"]).index(risks["store_id"].iloc[0]) if len(risks) else 0)
    products = q.focus_products(role, sku_store)
    first = risks.loc[risks["store_id"] == sku_store, "product_id"]
    default = list(products["product_id"]).index(first.iloc[0]) if len(first) else 0
    product = s2.selectbox("Focus SKU", products["product_id"], index=default,
                           format_func=dict(zip(products["product_id"], products["product_name"])).get)
    timeline_sku = q.sku_timeline(role, sku_store, product)
    st.plotly_chart(stock_chart(timeline_sku), width="stretch")
    notes("Stock line nearing the reorder line while sales hold = stockout coming",
          "Bars after today = forecast demand · whiskers = likely range")
    row = q.risk_list(role, [sku_store])
    row = row[row["product_id"] == product]
    with st.container(border=True):
        label("Key takeaways")
        st.markdown("\n".join(f"- {line}" for line in stock_summary(timeline_sku, row.iloc[0] if len(row) else None)))

    assumptions = q.replenishment_assumptions(role)
    with st.expander("How the suggested order is calculated"):
        lead, review = int(assumptions["lead_time_days"]), int(assumptions["review_days"])
        notes(f"**Suggested order** = forecast over {lead} + {review} days + safety stock − stock (rounded up, ≥ 0)",
              f"**Safety stock** = 1.65 × std(daily units, 28 days) × √{lead} "
              f"(≈ {assumptions['service_level']:.0%} service level)",
              f"**Lead time** = {lead} days — an assumption (suppliers deliver in 1–2 days)")
    st.subheader("Do the risk tiers work? (September backtest)")
    notes("Flag raised while in stock → did it run out within 3 days?")
    st.dataframe(q.stockout_backtest(role), hide_index=True, width="stretch",
                 column_config={"precision": st.column_config.NumberColumn(format="percent"),
                                "recall": st.column_config.NumberColumn(format="percent"),
                                "flag_rate": st.column_config.NumberColumn("flag rate", format="percent")})

# ------------------------------------------------------------------------------------------- explorer
def explorer() -> None:
    role = _open("explorer")
    tab_intro("What did empty shelves cost us, and why did a SKU run out?")
    notes("Every stock movement for one store × SKU", "Weekly counts vs calculated stock",
          "Stockout days and the sales they cost")
    lost_by_store = q.lost_sales_by_store(role)
    total_rev = float(lost_by_store["lost_revenue"].sum()) if len(lost_by_store) else 0.0
    total_units = float(lost_by_store["lost_units"].sum()) if len(lost_by_store) else 0.0
    e = st.columns(3)
    e[0].metric("Estimated lost sales (all stores)", f"₹{total_rev:,.0f}",
                help="Revenue focus SKUs would have made on their stockout days, before substitutes")
    e[1].metric("Estimated lost units", f"{total_units:,.0f}")
    e[2].metric("Stockout days", f"{int(lost_by_store['stockout_days'].sum()) if len(lost_by_store) else 0:,}",
                help="store × SKU days on which stock reached zero")
    top = q.top_lost_sales_skus(role, 10)
    left, right = st.columns([3, 2])
    left.markdown("**SKUs that lost the most sales**")
    left.dataframe(top.drop(columns=["store_id", "product_id"]), hide_index=True, width="stretch",
                   column_config={"lost_revenue": st.column_config.NumberColumn("lost revenue (₹)", format="%.0f")})
    right.markdown("**Lost sales by store**")
    right.dataframe(lost_by_store, hide_index=True, width="stretch",
                    column_config={"lost_revenue": st.column_config.NumberColumn("lost revenue (₹)", format="%.0f")})
    notes("**Lost units** = normal daily sales (in-stock days, last 28) − units sold, on stockout days",
          "**Lost revenue** = lost units × price · before substitutes (the store lost less overall)")
    st.warning("**How far to trust it**\n"
               "- Total and biggest losers: reliable (within 1% of the true figure)\n"
               "- Small losers (1–2 units): ranking not reliable\n"
               "- Cause: rebuilt daily stock misses ~⅓ of stockout days (docs/08)", icon=":material/warning:")

    st.subheader("One SKU in detail")
    stores = q.stores(role)
    names = dict(zip(stores["store_id"], stores["store_name"]))
    x1, x2 = st.columns(2)
    default_store = top["store_id"].iloc[0] if len(top) else stores["store_id"].iloc[0]
    ex_store = x1.selectbox("Store  ", stores["store_id"], format_func=names.get,
                            index=list(stores["store_id"]).index(default_store))
    ex_products = q.focus_products(role, ex_store)
    top_here = top.loc[top["store_id"] == ex_store, "product_id"]
    ex_default = list(ex_products["product_id"]).index(top_here.iloc[0]) if len(top_here) else 0
    ex_product = x2.selectbox("Focus SKU ", ex_products["product_id"], index=ex_default,
                              format_func=dict(zip(ex_products["product_id"], ex_products["product_name"])).get)
    s = q.explorer_summary(role, ex_store, ex_product)
    k = st.columns(5)
    k[0].metric("Stockout days", int(s["stockout_days"]))
    k[1].metric("Units sold", f"{int(s['units_sold']):,}")
    k[2].metric("Restocked / damaged", f"{int(s['restocked'])} / {int(s['damaged'])}")
    k[3].metric("Est. lost sales", f"₹{float(s['lost_revenue']):,.0f}", help=f"{float(s['lost_units']):.1f} units")
    k[4].metric("Avg |count − calculated|", f"{s['mean_abs_gap'] if s['mean_abs_gap'] == s['mean_abs_gap'] else 0} units",
                help=f"Average gap between the weekly count and the calculated stock over {int(s['counts_compared'])} "
                     "counts; non-zero when upstream rows were quarantined in Silver")
    timeline = q.explorer_timeline(role, ex_store, ex_product)
    st.plotly_chart(explorer_chart(timeline), width="stretch")
    with st.expander("Daily rows (latest first)"):
        st.dataframe(timeline.sort_values("day", ascending=False), hide_index=True, width="stretch", height=320)


PAGES = [(key, title, icon, globals()[key]) for key, title, icon in SECTIONS]

if __name__ == "__main__":   # run as a script (tests): every section on one page
    for *_, render in PAGES:
        render()
