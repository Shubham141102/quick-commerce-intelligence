"""Business & Growth workspace (persona consolidation, 2026-10-14): the former Business & Revenue (5B/5C) and
Customer Growth & Marketing (5E) workspaces merged into one persona with four pages. One question for the whole
workspace: "Are we growing profitably and keeping customers?"

    Performance    sales + delivery (one KPI row, one trend, one store scorecard)
    Alerts         anomaly detection (demand spikes, payment failures)
    Customers      segments + retention
    Growth levers  basket rules + recommendations + promotions

Every number comes from src/serving/queries.py (role-checked); formulas are unchanged from the former pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    MODEL,
    anomaly_chart,
    bar_chart,
    cohort_heatmap,
    delivery_trend_chart,
    revenue_trend_chart,
    segment_bubble_chart,
)
from app.components.insights import style_tiers  # noqa: E402
from app.components.ui import notes, page_header, require_workspace, tab_intro  # noqa: E402
from src.serving import queries as q  # noqa: E402

WS = "growth"
SECTIONS = [  # (function, sidebar title, icon) — each is a page in the sidebar
    ("performance", "Performance", ":material/payments:"),
    ("alerts", "Alerts", ":material/troubleshoot:"),
    ("customers", "Customers", ":material/groups:"),
    ("levers", "Growth levers", ":material/sell:"),
]
_TITLES = {key: title for key, title, _ in SECTIONS}
# The store-outage detector stays in the pipeline and the docs (experimental: ~1 order an hour makes a 3–6 hour
# outage indistinguishable from chance), but is no longer shown as an alert.
DETECTORS = {"demand_spike": "Demand spike", "payment_failure": "Payment failure"}


def _open(key: str) -> str:
    """Guard + header for one section page; returns the role."""
    role = require_workspace(WS)
    page_header(WS, _TITLES[key])
    return role


def _filters(role: str) -> tuple:
    """Period and store filters for Performance and Alerts. Kept in session state so they carry over between
    pages."""
    bounds = q.period_bounds(role)
    first, last = bounds["first_day"].date(), bounds["last_day"].date()
    stores = q.stores(role)
    names = dict(zip(stores["store_id"], stores["store_name"]))
    for key, default in (("gro_period", (first, last)), ("gro_stores", [])):
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"_{key}", default)
    f1, f2 = st.columns([1, 2])
    picked = f1.date_input("Period", min_value=first, max_value=last, key="gro_period")
    store_ids = f2.multiselect("Stores", stores["store_id"], format_func=names.get, placeholder="All stores",
                               key="gro_stores")
    st.session_state["_gro_period"], st.session_state["_gro_stores"] = picked, store_ids
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (first, last))
    return start, end, (store_ids or None)

# ------------------------------------------------------------------------------------------- performance
def performance(filters: tuple | None = None) -> None:
    role = _open("performance")
    start, end, store_ids = filters or _filters(role)
    tab_intro("Are we selling more, and are we keeping the delivery promise?")
    k = q.sales_kpis(role, start, end, store_ids)
    cur, prev = k["current"], k["previous"]
    d = q.delivery_kpis(role, start, end, store_ids)

    def delta(key: str, fmt: str = "{:+.1%}") -> str | None:
        a, b = cur.get(key), prev.get(key)
        if a is None or b in (None, 0) or prev["orders_placed"] == 0:
            return None
        return fmt.format(float(a) / float(b) - 1)

    c = st.columns(5)
    c[0].metric("Net revenue", f"₹{float(cur['net_revenue'] or 0):,.0f}", delta("net_revenue"),
                help="GMV − discount − completed refunds (see Metric definitions)")
    c[1].metric("Completed orders", f"{int(cur['orders_completed'] or 0):,}", delta("orders_completed"))
    c[2].metric("Average order value", f"₹{cur['aov']:,.0f}" if cur["aov"] else "—", delta("aov"),
                help="(GMV − discount) ÷ completed orders")
    c[3].metric("On-time rate", f"{d['on_time_rate']:.1%}" if d["on_time_rate"] is not None else "—",
                help="Delivered within 15 minutes of pickup ÷ delivered")
    rate = cur["cancellation_rate"]
    c[4].metric("Cancellation rate", f"{rate:.1%}" if rate is not None else "—",
                f"{(rate - prev['cancellation_rate']) * 100:+.1f} pts" if rate is not None and prev["cancellation_rate"] else None,
                delta_color="inverse")
    st.caption(f"Δ vs the previous {k['days']} days · blank when outside the data")

    grain = st.radio("Revenue trend by", ["day", "week"], horizontal=True, index=1)
    st.plotly_chart(revenue_trend_chart(q.sales_trend(role, start, end, store_ids, grain)), width="stretch")
    st.markdown("**Store scorecard** (highest revenue first)")
    st.dataframe(q.store_scorecard(role, start, end, store_ids), hide_index=True, width="stretch",
                 column_config={"net_revenue": st.column_config.NumberColumn("net revenue (₹)", format="%.0f"),
                                "orders_completed": "orders",
                                "aov": st.column_config.NumberColumn("AOV (₹)", format="%.0f"),
                                "on_time_rate": st.column_config.NumberColumn("on-time", format="percent"),
                                "avg_minutes": st.column_config.NumberColumn("avg delivery (min)", format="%.1f"),
                                "cancellation_rate": st.column_config.NumberColumn("cancel rate", format="percent")})
    left, right = st.columns(2)
    cats = q.category_contribution(role, start, end, store_ids)
    left.markdown("**Category contribution**")
    left.plotly_chart(bar_chart(cats.head(12), "revenue", "category", x_title="revenue (₹)"), width="stretch")
    right.markdown("**Cancellations by reason**")
    right.dataframe(q.cancellations_by_reason(role, start, end, store_ids), hide_index=True, width="stretch")
    st.markdown("**Top products** (all stores, months overlapping the period)")
    st.dataframe(q.top_products(role, start, end), hide_index=True, width="stretch",
                 column_config={"revenue": st.column_config.NumberColumn("revenue (₹)", format="%.0f")})
    with st.expander("Delivery detail"):
        e = st.columns(4)
        e[0].metric("Avg delivery time", f"{d['avg_minutes']:.1f} min" if d["avg_minutes"] is not None else "—")
        e[1].metric("Typical 90th percentile", f"{d['typical_p90']:.1f} min" if d["typical_p90"] is not None else "—",
                    help="Median over store-days of the 90th-percentile delivery time")
        e[2].metric("Failed deliveries", f"{int(d['failed'] or 0):,}")
        e[3].metric("Cancelled in transit", f"{int(d['cancelled_in_transit'] or 0):,}")
        st.plotly_chart(delivery_trend_chart(q.delivery_trend(role, start, end, store_ids)), width="stretch")
    with st.expander("Metric definitions"):
        notes("**Net revenue** — GMV − discount − completed refunds",
              "**GMV** — line revenue of completed orders",
              "**AOV** — (GMV − discount) ÷ completed orders",
              "**Completed** — delivered · paid · not cancelled",
              "**Cancellations** — before dispatch mostly customer-side; after dispatch delivery-side",
              "Full list: docs/metric_definitions.md")

# ------------------------------------------------------------------------------------------- alerts
def alerts(filters: tuple | None = None) -> None:
    role = _open("alerts")
    start, end, store_ids = filters or _filters(role)
    tab_intro("Did something unusual happen that needs investigating?")
    st.info("**Signal to investigate, not proof** — compared with the last 28 days for that store, hour or product.",
            icon=":material/search:")
    summary = q.anomaly_summary(role)
    cols = st.columns(len(DETECTORS))
    for col, det in zip(cols, DETECTORS):
        row = summary[summary["detector"] == det]
        col.metric(DETECTORS[det], int(row["anomalies"].iloc[0]) if len(row) else 0,
                   help={"demand_spike": "A product ordered far more than normal at a store",
                         "payment_failure": "Failed payments and payment errors far above normal"}[det])
    a1, a2 = st.columns(2)
    dets = a1.multiselect("Type", list(DETECTORS), format_func=DETECTORS.get, placeholder="All types")
    found = q.anomaly_list(role, dets or list(DETECTORS), store_ids)
    a2.caption(f"{len(found)} anomalies · store filter applies · payment failures are network-wide")
    st.dataframe(style_tiers(found.drop(columns=["observed", "expected"])
                             .assign(detector=lambda x: x["detector"].map(DETECTORS)), ["severity"]),
                 hide_index=True, width="stretch",
                 column_config={"score": st.column_config.NumberColumn("score", help="−log10(p): higher = more unusual",
                                                                       format="%.1f")})
    if len(found):
        pick = st.selectbox("Inspect an anomaly", found["anomaly_id"],
                            format_func=lambda a: f"{a} — {found.set_index('anomaly_id').loc[a, 'description'][:90]}")
        st.plotly_chart(anomaly_chart(q.anomaly_series(role, pick)), width="stretch")
    with st.expander("How the detectors work and their limits"):
        notes("**Demand spike** — product orders vs its usual share of store orders (p < 0.0001)",
              "**Payment failure** — failed payments + payment ERROR logs per hour vs normal",
              "**Method** — negative binomial: allows for weekday, rain and promotion swings",
              "**Store outages** — not shown: at ~1 order/hour a short outage looks like chance (experimental)",
              "Accuracy vs planted anomalies: docs/09_business_workspace.md")

# ------------------------------------------------------------------------------------------- customers
def customers() -> None:
    role = _open("customers")
    tab_intro("Who are our customers, and who is worth winning back?")
    k = q.marketing_kpis(role)
    c = st.columns(5)
    c[0].metric("Customers", f"{int(k['customers']):,}")
    c[1].metric("Have ordered", f"{int(k['buyers']):,}")
    c[2].metric("Repeat rate", f"{k['repeat_rate']:.1%}" if k["repeat_rate"] is not None else "—",
                help="Customers with 2+ completed orders ÷ customers with 1+")
    c[3].metric("Active (≤ 14 days)", f"{int(k['active']):,}")
    c[4].metric("At risk or lapsed", f"{int(k['at_risk_or_lapsed']):,}",
                help="No completed order for more than 30 days")

    st.subheader("Segments")
    prof = q.segment_profiles(role)
    notes(f"{len(prof)} segments · K-means on spend, frequency, timing, promotions, category mix",
          "Names from measured traits · campaign ideas are untested suggestions")
    st.plotly_chart(segment_bubble_chart(prof), width="stretch")
    st.dataframe(prof, hide_index=True, width="stretch",
                 column_config={"share": st.column_config.NumberColumn("share", format="percent"),
                                "avg_spend": st.column_config.NumberColumn("avg spend (₹)", format="%.0f"),
                                "avg_order_value": st.column_config.NumberColumn("AOV (₹)", format="%.0f"),
                                "night_order_share": st.column_config.NumberColumn("night", format="percent"),
                                "weekend_order_share": st.column_config.NumberColumn("weekend", format="percent"),
                                "promo_order_share": st.column_config.NumberColumn("promo", format="percent")})
    labels = dict(zip(prof["segment_id"], prof["segment_label"]))
    with st.expander("Customers in a segment"):
        pick = st.selectbox("Segment", list(labels), format_func=labels.get, key="seg_pick")
        st.dataframe(q.segment_customers(role, pick), hide_index=True, width="stretch",
                     column_config={"total_spend": st.column_config.NumberColumn("spend (₹)", format="%.0f"),
                                    "avg_order_value": st.column_config.NumberColumn("AOV (₹)", format="%.0f")})

    st.subheader("Retention and win-back")
    st.caption("Descriptive only — no churn prediction (no reliable churn signal).")
    status = q.retention_status(role)
    notes("Active ≤ 14 days · Cooling 15–30 · At risk 31–60 · Lapsed > 60 (since last order)")
    grid = status.pivot_table(index="status", columns="value_tier", values="customers", fill_value=0, sort=False)
    st.dataframe(grid, width="stretch")
    f1, f2, f3 = st.columns([2, 2, 1])
    sts = f1.multiselect("Status", ["active", "cooling", "at_risk", "lapsed"], default=["at_risk", "lapsed"])
    tiers = f2.multiselect("Value tier", ["High", "Medium", "Low"], default=["High"])
    overdue = f3.checkbox("Overdue only", help="Days since last order > 2 × their own average gap")
    who = q.retention_customers(role, sts or None, tiers or None, overdue)
    st.caption(f"{len(who)} customers · highest spend first · win-back list")
    st.dataframe(who, hide_index=True, width="stretch",
                 column_config={"total_spend": st.column_config.NumberColumn("spend (₹)", format="%.0f")})
    with st.expander("Cohort retention"):
        notes("Row = month of first order · cell = share ordering again N months later")
        st.plotly_chart(cohort_heatmap(q.retention_cohorts(role)), width="stretch")
    with st.expander("How the number of segments was chosen"):
        st.dataframe(q.segmentation_selection(role), hide_index=True, width="stretch")
        notes("**k** — best silhouette (how well separated)",
              "**Stability** — agreement with 5 reruns (1 = identical)")

# ------------------------------------------------------------------------------------------- growth levers
def levers() -> None:
    role = _open("levers")
    tab_intro("What can we do to sell more: bundles, personal suggestions and promotions?")

    st.subheader("Bought together")
    notes("**A → B** — baskets with A often contain B",
          "**Confidence** — share of A-baskets with B · **Lift** — × more than chance (1 = no link)",
          "Use: shelve together · bundle · suggest B at checkout")
    products = q.rule_products(role)
    names = dict(zip(products["product_id"], products["product_name"]))
    left, right = st.columns([1, 1])
    with left:
        prod = st.selectbox("What is bought with…", list(names), format_func=names.get, index=None,
                            placeholder="Pick a product")
        if prod:
            st.dataframe(q.product_partners(role, prod), hide_index=True, width="stretch",
                         column_config={"confidence": st.column_config.NumberColumn("confidence", format="percent")})
    with right:
        cats = q.categories(role)
        cat_names = dict(zip(cats["category_id"], cats["category_name"]))
        chosen = st.multiselect("Antecedent category", list(cat_names), format_func=cat_names.get,
                                placeholder="All categories")
        min_lift = st.slider("Minimum lift", 2.0, 20.0, 2.0, 0.5)
    rules = q.basket_rules(role, chosen or None, min_lift)
    st.markdown(f"**Strongest rules** ({len(rules)} shown · confidence ≥ 10%, lift ≥ 2)")
    st.dataframe(rules, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.NumberColumn("confidence", format="percent"),
                                "lift": st.column_config.NumberColumn("lift", format="%.1f")})

    st.subheader("Personal recommendations")
    notes("Top 10 per customer = buy-again + often-bought-with + best-sellers")
    prof = q.segment_profiles(role)
    labels = dict(zip(prof["segment_id"], prof["segment_label"]))
    s1, s2 = st.columns(2)
    seg = s1.selectbox("Segment", [None, *labels], format_func=lambda s: "All segments" if s is None else labels[s],
                       key="rec_seg")
    people = q.recommendation_customers(role, seg)
    seg_of = dict(zip(people["customer_id"], people["segment_label"]))
    cust = s2.selectbox("Customer (highest spend first)", list(seg_of), format_func=lambda cid: f"{cid} — {seg_of[cid]}")
    if cust:
        st.dataframe(q.customer_recommendations(role, cust), hide_index=True, width="stretch")
    if seg is not None:
        st.markdown(f"**New-to-them products most recommended in “{labels[seg]}”** (campaign candidates)")
        top = q.segment_top_recommendations(role, seg)
        if len(top):
            st.plotly_chart(bar_chart(top, "customers", "product", color=MODEL, x_title="customers"), width="stretch")
    with st.expander("How good are the recommendations? (September test)"):
        m = q.recommendation_metrics(role).set_index("method")
        c = st.columns(3)
        for col, method, title in zip(c, ("hybrid", "repeat", "popularity"),
                                      ("Hybrid (used)", "Buy-again only", "Most popular")):
            if method in m.index:
                col.metric(title, f"{m.loc[method, 'precision_at_10']:.1%}",
                           help=f"Precision@10; hit rate {m.loc[method, 'hit_rate']:.0%}, "
                                f"recall@10 {m.loc[method, 'recall_at_10']:.1%}, coverage {m.loc[method, 'coverage']:.0%}")
        notes("**Precision@10** — share of the 10 suggestions actually bought",
              "Weights chosen on Aug · trained Apr–Aug · tested once on Sept purchases",
              "Only ~¼ of purchases are repeats → best-sellers are a strong baseline")
        st.dataframe(m.reset_index(), hide_index=True, width="stretch")

    st.subheader("Promotions")
    pm = q.promotion_metrics(role)
    notes("**Uplift** — category daily units during the promotion vs no-promotion days",
          "Before/after comparison — not a controlled test")
    if len(pm):
        c = st.columns(3)
        c[0].metric("Promotions", len(pm))
        c[1].metric("Total discount cost", f"₹{float(pm['discount_cost'].sum()):,.0f}")
        c[2].metric("Median uplift", f"{pm['uplift_pct'].median():+.1f}%")
        st.plotly_chart(bar_chart(pm.assign(label=pm["name"] + " (" + pm["category"].fillna("") + ")"),
                                  "uplift_pct", "label", color=MODEL, x_title="uplift %"), width="stretch")
    with st.expander("All promotions"):
        st.dataframe(pm, hide_index=True, width="stretch",
                     column_config={"discount_cost": st.column_config.NumberColumn("discount cost (₹)", format="%.0f"),
                                    "uplift_pct": st.column_config.NumberColumn("uplift %", format="%.1f")})


PAGES = [(key, title, icon, globals()[key]) for key, title, icon in SECTIONS]

if __name__ == "__main__":   # run as a script (tests): filters once, then every section on one page
    _filters_once = _filters(require_workspace(WS))
    for key, *_, render in PAGES:
        if key in ("performance", "alerts"):
            render(_filters_once)
        else:
            render()
