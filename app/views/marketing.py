"""Customer Growth & Marketing workspace (Phase 5E): segments, basket & affinity, recommendations, retention
(descriptive) and promotion effectiveness. Every number comes from src/serving/queries.py (role-checked)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components.charts import (  # noqa: E402
    MODEL,
    bar_chart,
    cohort_heatmap,
    segment_bubble_chart,
)
from app.components.ui import notes, page_header, require_workspace, tab_intro  # noqa: E402
from src.serving import queries as q  # noqa: E402

WS = "marketing"
SECTIONS = [  # (function, sidebar title, icon) — each is a page in the sidebar
    ("segments", "Segments", ":material/groups:"),
    ("basket", "Basket & affinity", ":material/shopping_basket:"),
    ("recs", "Recommendations", ":material/recommend:"),
    ("retention", "Retention", ":material/autorenew:"),
    ("promos", "Promotion effectiveness", ":material/sell:"),
]
_TITLES = {key: title for key, title, _ in SECTIONS}


def _open(key: str) -> str:
    """Guard + header for one section page; returns the role."""
    role = require_workspace(WS)
    page_header(WS, _TITLES[key])
    return role


def _kpis(role: str) -> None:
    """Customer KPI strip (shown on the first Marketing page)."""
    k = q.marketing_kpis(role)
    c = st.columns(5)
    c[0].metric("Customers", f"{int(k['customers']):,}")
    c[1].metric("Have ordered", f"{int(k['buyers']):,}")
    c[2].metric("Repeat rate", f"{k['repeat_rate']:.1%}" if k["repeat_rate"] is not None else "—",
                help="Customers with 2+ completed orders ÷ customers with 1+")
    c[3].metric("Active (≤ 14 days)", f"{int(k['active']):,}")
    c[4].metric("At risk or lapsed", f"{int(k['at_risk_or_lapsed']):,}",
                help="No completed order for more than 30 days")

# ------------------------------------------------------------------------------------------- segments
def segments() -> None:
    role = _open("segments")
    tab_intro("Who are our customers, and how should we talk to each group?")
    _kpis(role)
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
    pick = st.selectbox("Customers in segment", list(labels), format_func=labels.get, key="seg_pick")
    st.dataframe(q.segment_customers(role, pick), hide_index=True, width="stretch",
                 column_config={"total_spend": st.column_config.NumberColumn("spend (₹)", format="%.0f"),
                                "avg_order_value": st.column_config.NumberColumn("AOV (₹)", format="%.0f")})
    with st.expander("How the number of segments was chosen"):
        st.dataframe(q.segmentation_selection(role), hide_index=True, width="stretch")
        notes("**k** — best silhouette (how well separated)",
              "**Stability** — agreement with 5 reruns (1 = identical)")

# ------------------------------------------------------------------------------------------- basket
def basket() -> None:
    role = _open("basket")
    tab_intro("What do customers buy together?")
    notes("**A → B** — baskets with A often contain B",
          "**Confidence** — share of A-baskets with B · **Lift** — × more than chance (1 = no link)",
          "Kept: confidence ≥ 10%, lift ≥ 2")
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
    st.markdown(f"**Strongest rules** ({len(rules)} shown)")
    st.dataframe(rules, hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.NumberColumn("confidence", format="percent"),
                                "lift": st.column_config.NumberColumn("lift", format="%.1f")})
    notes("Use: shelve together · bundle · suggest B at checkout")

# ------------------------------------------------------------------------------------------- recommendations
def recs() -> None:
    role = _open("recs")
    tab_intro("What should we show each customer next, and does it work?")
    m = q.recommendation_metrics(role).set_index("method")
    st.markdown("**How good are the recommendations?**")
    notes("Weights chosen on Aug · trained Apr–Aug · tested once on Sept purchases")
    c = st.columns(3)
    for col, method, title in zip(c, ("hybrid", "repeat", "popularity"),
                                  ("Hybrid (used)", "Buy-again only", "Most popular")):
        if method in m.index:
            col.metric(title, f"{m.loc[method, 'precision_at_10']:.1%}",
                       help=f"Precision@10; hit rate {m.loc[method, 'hit_rate']:.0%}, "
                            f"recall@10 {m.loc[method, 'recall_at_10']:.1%}, coverage {m.loc[method, 'coverage']:.0%}")
    if "hybrid" in m.index:
        st.caption(f"Hybrid weights: {m.loc['hybrid', 'weights']}.")
    notes("**Precision@10** — share of the 10 suggestions actually bought",
          "Only ~¼ of purchases are repeats → best-sellers are a strong baseline",
          "Hybrid = buy-again + often-bought-with + best-sellers")
    st.dataframe(m.reset_index(), hide_index=True, width="stretch")
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

# ------------------------------------------------------------------------------------------- retention
def retention() -> None:
    role = _open("retention")
    tab_intro("Who is still ordering, and who is worth winning back?")
    st.info("**Descriptive only** — no churn prediction (no reliable churn signal; Tier 2 item).",
            icon=":material/info:")
    st.markdown("**Cohort retention**")
    notes("Row = month of first order · cell = share ordering again N months later")
    st.plotly_chart(cohort_heatmap(q.retention_cohorts(role)), width="stretch")
    status = q.retention_status(role)
    st.markdown("**Status at end of data**")
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

# ------------------------------------------------------------------------------------------- promotions
def promos() -> None:
    role = _open("promos")
    tab_intro("Did our promotions lift sales, and what did they cost?")
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
    st.dataframe(pm, hide_index=True, width="stretch",
                 column_config={"discount_cost": st.column_config.NumberColumn("discount cost (₹)", format="%.0f"),
                                "uplift_pct": st.column_config.NumberColumn("uplift %", format="%.1f")})


PAGES = [(key, title, icon, globals()[key]) for key, title, icon in SECTIONS]

if __name__ == "__main__":   # run as a script (tests): every section on one page
    for *_, render in PAGES:
        render()
