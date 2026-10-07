"""Business & Revenue workspace (Phase 5B/5C): sales & revenue performance, delivery & operations
(and anomaly detection, added in 5C). Every number comes from src/serving/queries.py (role-checked)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    anomaly_chart,
    bar_chart,
    delivery_trend_chart,
    revenue_trend_chart,
)
from app.components.insights import style_tiers  # noqa: E402
from app.components.ui import page_header, require_workspace, tab_intro  # noqa: E402
from src.serving import queries as q  # noqa: E402

WS = "business"
SECTIONS = [  # (function, sidebar title, icon) — each is a page in the sidebar
    ("sales", "Sales & revenue performance", ":material/payments:"),
    ("delivery", "Delivery & operations", ":material/local_shipping:"),
    ("anomalies", "Anomaly detection", ":material/troubleshoot:"),
]
_TITLES = {key: title for key, title, _ in SECTIONS}


def _open(key: str) -> str:
    """Guard + header for one section page; returns the role."""
    role = require_workspace(WS)
    page_header(WS, _TITLES[key])
    return role


def _filters(role: str) -> tuple:
    """Period and store filters, shown on every Business page. Values are kept in session state so they carry
    over when moving between the Business pages."""
    bounds = q.period_bounds(role)
    first, last = bounds["first_day"].date(), bounds["last_day"].date()
    stores = q.stores(role)
    names = dict(zip(stores["store_id"], stores["store_name"]))
    for key, default in (("biz_period", (first, last)), ("biz_stores", [])):
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"_{key}", default)
    f1, f2 = st.columns([1, 2])
    picked = f1.date_input("Period", min_value=first, max_value=last, key="biz_period")
    store_ids = f2.multiselect("Stores", stores["store_id"], format_func=names.get, placeholder="All stores",
                               key="biz_stores")
    st.session_state["_biz_period"], st.session_state["_biz_stores"] = picked, store_ids
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (first, last))
    return start, end, (store_ids or None)

# ------------------------------------------------------------------------------------------- sales
def sales(filters: tuple | None = None) -> None:
    role = _open("sales")
    start, end, store_ids = filters or _filters(role)
    tab_intro("How are we selling compared with the previous period, and where does revenue come from?")
    k = q.sales_kpis(role, start, end, store_ids)
    cur, prev = k["current"], k["previous"]

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
    c[3].metric("Units sold", f"{int(cur['units'] or 0):,}", delta("units"))
    rate = cur["cancellation_rate"]
    c[4].metric("Cancellation rate", f"{rate:.1%}" if rate is not None else "—",
                f"{(rate - prev['cancellation_rate']) * 100:+.1f} pts" if rate is not None and prev["cancellation_rate"] else None,
                delta_color="inverse")
    st.caption(f"Changes compare with the {k['days']} days before the selected period (empty when that period is "
               "outside the data).")

    grain = st.radio("Trend by", ["day", "week"], horizontal=True, index=1)
    st.plotly_chart(revenue_trend_chart(q.sales_trend(role, start, end, store_ids, grain)), width="stretch")
    left, right = st.columns(2)
    left.markdown("**Revenue by store**")
    left.dataframe(q.revenue_by_store(role, start, end, store_ids), hide_index=True, width="stretch",
                   column_config={"net_revenue": st.column_config.NumberColumn("net revenue (₹)", format="%.0f"),
                                  "aov": st.column_config.NumberColumn("AOV (₹)", format="%.0f"),
                                  "cancellation_rate": st.column_config.NumberColumn("cancel rate", format="percent")})
    cats = q.category_contribution(role, start, end, store_ids)
    right.markdown("**Category contribution**")
    right.plotly_chart(bar_chart(cats.head(12), "revenue", "category", x_title="revenue (₹)"), width="stretch")
    st.markdown("**Top products** (all stores, months overlapping the period)")
    st.dataframe(q.top_products(role, start, end), hide_index=True, width="stretch",
                 column_config={"revenue": st.column_config.NumberColumn("revenue (₹)", format="%.0f")})
    with st.expander("Metric definitions"):
        st.markdown("- **Net revenue** = GMV − discount − completed refunds (refunds on the refund's date)\n"
                    "- **GMV** = line revenue of completed orders (catalog price for lines with a flagged unit price)\n"
                    "- **AOV** = (GMV − discount) ÷ completed orders\n"
                    "- **Completed order** = delivered, successful payment, not cancelled\n"
                    "- Full list: docs/metric_definitions.md")

# ------------------------------------------------------------------------------------------- delivery
def delivery(filters: tuple | None = None) -> None:
    role = _open("delivery")
    start, end, store_ids = filters or _filters(role)
    tab_intro("Are we keeping the delivery promise, and why are orders cancelled?")
    d = q.delivery_kpis(role, start, end, store_ids)
    c = st.columns(5)
    c[0].metric("On-time rate", f"{d['on_time_rate']:.1%}" if d["on_time_rate"] is not None else "—",
                help="Delivered within 15 minutes of pickup ÷ delivered")
    c[1].metric("Avg delivery time", f"{d['avg_minutes']:.1f} min" if d["avg_minutes"] is not None else "—")
    c[2].metric("Typical 90th percentile", f"{d['typical_p90']:.1f} min" if d["typical_p90"] is not None else "—",
                help="Median over store-days of the 90th-percentile delivery time")
    c[3].metric("Failed deliveries", f"{int(d['failed'] or 0):,}")
    c[4].metric("Cancelled in transit", f"{int(d['cancelled_in_transit'] or 0):,}")
    st.plotly_chart(delivery_trend_chart(q.delivery_trend(role, start, end, store_ids)), width="stretch")
    left, right = st.columns(2)
    left.markdown("**By store** (worst on-time rate first)")
    left.dataframe(q.delivery_by_store(role, start, end, store_ids), hide_index=True, width="stretch",
                   column_config={"on_time_rate": st.column_config.NumberColumn("on-time", format="percent")})
    canc = q.cancellations_by_reason(role, start, end, store_ids)
    right.markdown("**Cancellations by reason**")
    right.dataframe(canc, hide_index=True, width="stretch")
    st.caption("Cancellations before dispatch are mostly customer-side (changed mind, item unavailable, payment); after "
               "dispatch they are delivery-side (delays, unreachable customer, address). Weekly figures: periods are "
               "matched by week.")

# ------------------------------------------------------------------------------------------- anomalies
def anomalies(filters: tuple | None = None) -> None:
    role = _open("anomalies")
    start, end, store_ids = filters or _filters(role)
    tab_intro("Did something unusual happen that needs investigating?")
    st.info("An anomaly is a **signal to investigate**, not proof of fraud or failure. Each one compares what happened "
            "with what is normal for that store, hour or product over the previous 28 days.", icon=":material/search:")
    summary = q.anomaly_summary(role)
    labels = {"store_outage": "Store outage (experimental)", "demand_spike": "Demand spike",
              "payment_failure": "Payment failure"}
    cols = st.columns(3)
    for col, det in zip(cols, labels):
        row = summary[summary["detector"] == det]
        col.metric(labels[det], int(row["anomalies"].iloc[0]) if len(row) else 0,
                   help={"store_outage": "Hours with far fewer orders than normal at a store",
                         "demand_spike": "A product ordered far more than normal at a store",
                         "payment_failure": "Failed payments and payment errors far above normal"}[det])
    a1, a2 = st.columns(2)
    dets = a1.multiselect("Type", list(labels), format_func=labels.get, placeholder="All types")
    found = q.anomaly_list(role, dets or None, store_ids)
    a2.caption(f"{len(found)} anomalies (store filter above applies; payment failures are network-wide).")
    st.dataframe(style_tiers(found.drop(columns=["observed", "expected"])
                             .assign(detector=lambda d: d["detector"].map(labels)), ["severity"]),
                 hide_index=True, width="stretch",
                 column_config={"score": st.column_config.NumberColumn("score", help="−log10(p): higher = more unusual",
                                                                       format="%.1f")})
    if len(found):
        pick = st.selectbox("Inspect an anomaly", found["anomaly_id"],
                            format_func=lambda a: f"{a} — {found.set_index('anomaly_id').loc[a, 'description'][:90]}")
        st.plotly_chart(anomaly_chart(q.anomaly_series(role, pick)), width="stretch")
    with st.expander("How the detectors work and their limits"):
        st.markdown(
            "- **Demand spike:** orders containing a product at a store on a day, vs its share of the store's orders "
            "over the last 28 days × that day's store orders (p < 0.0001).\n"
            "- Daily order counts swing more than pure chance would (weekdays, rain, promotions), so the p-values allow "
            "for that extra variation (negative binomial).\n"
            "- **Payment failure:** network-wide failed payments and payment-service ERROR logs per hour vs normal.\n"
            "- **Store outage (experimental):** runs of 2–8 hours with far fewer orders than normal. A store gets only ~1 order an hour, "
            "so short outages look like chance and are mostly **not detectable** from orders alone; real systems use "
            "store-system heartbeats, which this data does not have.\n"
            "- Accuracy against the planted anomalies: docs/09_business_workspace.md.")


PAGES = [(key, title, icon, globals()[key]) for key, title, icon in SECTIONS]

if __name__ == "__main__":   # run as a script (tests): filters once, then every section on one page
    _filters_once = _filters(require_workspace(WS))
    for *_, render in PAGES:
        render(_filters_once)
