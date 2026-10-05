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
from app.components.ui import advisory_note, freshness_banner, require_workspace  # noqa: E402
from src.serving import queries as q  # noqa: E402

role = require_workspace("inventory")
st.title("📦 Inventory & Supply Chain")
freshness_banner()
overview, forecasting, stockout, explorer = st.tabs(
    ["Overview", "Demand forecasting", "Stockout risk & replenishment", "Inventory explorer"])

# ------------------------------------------------------------------------------------------- overview
with overview:
    k = q.inventory_kpis(role)
    st.caption(f"Stock position at the end of **{k['as_of']:%d %b %Y}** for {k['skus']} store × focus-SKU pairs.")
    c = st.columns(5)
    c[0].metric("High risk", int(k["high"]), help="Stock is 0, or forecast demand over lead time + 1 day ≥ stock")
    c[1].metric("Medium risk", int(k["medium"]), help="Stock ≤ reorder level, or under 3 days of inventory")
    c[2].metric("Zero stock", int(k["zero_stock"]))
    c[3].metric("SKUs to reorder", k["skus_to_order"], help=f"{k['units_to_order']} units in total (advisory)")
    c[4].metric("Forecast, next 7 days", f"{k['forecast_units_7d']:,.0f} units",
                help=f"All stores and categories, {k['forecast_first_day']:%d %b}–{k['forecast_last_day']:%d %b}")
    st.subheader("Stock health by store")
    st.dataframe(q.stock_health_by_store(role), hide_index=True, width="stretch",
                 column_config={"median_days_of_cover": st.column_config.NumberColumn("median days of cover",
                                help="closing stock ÷ forecast daily demand")})
    advisory_note()

# ------------------------------------------------------------------------------------------- forecasting
with forecasting:
    stores, cats = q.stores(role), q.categories(role)
    a, b = st.columns(2)
    store = a.selectbox("Store", stores["store_id"], format_func=dict(zip(stores["store_id"], stores["store_name"])).get)
    cat = b.selectbox("Category", cats["category_id"], format_func=dict(zip(cats["category_id"], cats["category_name"])).get)
    st.plotly_chart(forecast_chart(q.category_forecast(role, store, cat)), width="stretch")
    st.caption("Blue: units actually sold (completed orders). Orange dotted: what the model (trained on Apr–Aug) "
               "forecast one day ahead during September. Red: the forecast for the 7 days after the data ends, "
               "with its 10–90% range.")

    acc = q.forecast_accuracy(role)
    cat_acc = acc[acc["evaluation"] == "category_daily"]
    pooled = cat_acc[cat_acc["horizon"] == 0].set_index("model")["wape"]
    st.subheader("How accurate is it? (September backtest)")
    m = st.columns(3)
    m[0].metric("Model WAPE", f"{pooled['model']:.3f}")
    m[1].metric("Moving average", f"{pooled['baseline_moving_avg']:.3f}",
                delta=f"{pooled['baseline_moving_avg'] - pooled['model']:+.3f} worse", delta_color="off")
    m[2].metric("Seasonal naive", f"{pooled['baseline_seasonal_naive']:.3f}",
                delta=f"{pooled['baseline_seasonal_naive'] - pooled['model']:+.3f} worse", delta_color="off")
    by_h = (cat_acc[cat_acc["horizon"] > 0].pivot_table(index="horizon", columns="model", values="wape")
            .rename(columns={"model": "model", "baseline_moving_avg": "moving average",
                             "baseline_seasonal_naive": "seasonal naive"}).round(3))
    left, right = st.columns(2)
    left.markdown("**WAPE by days ahead** (lower is better)")
    left.dataframe(by_h, width="stretch")
    right.markdown("**WAPE by category** (all horizons)")
    right.dataframe(q.accuracy_by_category(role).round(3), hide_index=True, width="stretch", height=260)
    st.markdown("**What drives the forecast** (permutation importance)")
    st.plotly_chart(importance_chart(q.feature_importance(role)), width="stretch")
    with st.expander("Limitations and how to read WAPE"):
        st.markdown(
            "- WAPE = total absolute error ÷ total units sold. A store-category sells only ~3 units a day, so much of "
            "the error is randomness no model can remove: a *perfect* model would still score about 0.58 on this "
            "data (see docs/05_demand_forecasting.md).\n"
            "- Future weather is assumed equal to the last 7-day average; no promotions are assumed after the data ends.\n"
            "- Holiday dates are approximate and the synthetic data has no holiday effect.\n"
            "- The model slightly under-forecasts (about −7%).")

# ------------------------------------------------------------------------------------------- stockout
with stockout:
    advisory_note()
    stores = q.stores(role)
    names = dict(zip(stores["store_id"], stores["store_name"]))
    f1, f2 = st.columns([2, 1])
    picked_stores = f1.multiselect("Stores", stores["store_id"], format_func=names.get, placeholder="All stores")
    tiers = f2.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    risks = q.risk_list(role, picked_stores or None, tiers or None)
    st.markdown(f"**{len(risks)} SKUs** — sorted by tier, then fewest days of cover")
    st.dataframe(risks.drop(columns=["store_id", "product_id"]), hide_index=True, width="stretch",
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
    st.plotly_chart(stock_chart(q.sku_timeline(role, sku_store, product)), width="stretch")

    assumptions = q.replenishment_assumptions(role)
    with st.expander("How the suggested order is calculated"):
        st.markdown(
            f"`suggested = ceil(max(0, forecast demand over {int(assumptions['lead_time_days'])} + "
            f"{int(assumptions['review_days'])} days + safety stock − current stock))`  \n"
            f"`safety stock = 1.65 × std(daily units, last 28 days) × √{int(assumptions['lead_time_days'])}` "
            f"(≈ {assumptions['service_level']:.0%} service level)  \n"
            f"Lead time of {int(assumptions['lead_time_days'])} days is an **assumption** (the simulated suppliers "
            "deliver in 1–2 days).")
    st.subheader("Do the risk tiers work? (September backtest)")
    st.caption("Decisions taken while the SKU was still in stock: did it run out within the next 3 days?")
    st.dataframe(q.stockout_backtest(role), hide_index=True, width="stretch",
                 column_config={"precision": st.column_config.NumberColumn(format="percent"),
                                "recall": st.column_config.NumberColumn(format="percent"),
                                "flag_rate": st.column_config.NumberColumn("flag rate", format="percent")})

# ------------------------------------------------------------------------------------------- explorer
with explorer:
    st.markdown("Drill into one store × focus SKU: every stock movement, the weekly counts against the calculated "
                "stock, stockout days and the sales they cost.")
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
    st.caption("Estimate = average daily units on the SKU's in-stock days in the previous 28 days − units sold, "
               "on stockout days, × catalog price. In the simulation shoppers often bought a substitute, so the store "
               "as a whole lost less than this.")
    st.warning("**How far to trust this:** the total and the biggest losers are reliable (total within 1% of the "
               "simulator's true lost units). Rankings among SKUs that lost only a unit or two are not: stockout days "
               "come from the rebuilt daily stock, which misses about a third of real stockout days. "
               "Details: docs/08_inventory_explorer_lost_sales.md.", icon="⚠️")

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
