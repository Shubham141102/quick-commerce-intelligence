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
from app.components.ui import page_header, require_workspace, tab_intro  # noqa: E402
from src.serving import queries as q  # noqa: E402

role = require_workspace("marketing")
page_header("marketing")

k = q.marketing_kpis(role)
c = st.columns(5)
c[0].metric("Customers", f"{int(k['customers']):,}")
c[1].metric("Have ordered", f"{int(k['buyers']):,}")
c[2].metric("Repeat rate", f"{k['repeat_rate']:.1%}" if k["repeat_rate"] is not None else "—",
            help="Customers with 2+ completed orders ÷ customers with 1+")
c[3].metric("Active (≤ 14 days)", f"{int(k['active']):,}")
c[4].metric("At risk or lapsed", f"{int(k['at_risk_or_lapsed']):,}", help="No completed order for more than 30 days")

segments, basket, recs, retention, promos = st.tabs(
    ["Segments", "Basket & affinity", "Recommendations", "Retention", "Promotion effectiveness"])

# ------------------------------------------------------------------------------------------- segments
with segments:
    tab_intro("Who are our customers, and how should we talk to each group?")
    prof = q.segment_profiles(role)
    st.caption(f"{len(prof)} segments found by K-means on spending, frequency, timing, promotion use and category mix. "
               "Names are generated from each segment's measured traits; campaign ideas are suggestions, not tested results.")
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
        st.markdown("k is the value with the best **silhouette** (how well separated the segments are). **Stability** "
                    "is the agreement (adjusted Rand index) with 5 reruns using different random seeds: near 1 means the "
                    "same customers land together every time.")

# ------------------------------------------------------------------------------------------- basket
with basket:
    tab_intro("What do customers buy together?")
    st.caption("Rules **A → B**: when a basket has A it often has B too. Confidence = share of A-baskets that also "
               "have B; lift = how many times more often than chance (1 = no link). Only rules with confidence ≥ 10% "
               "and lift ≥ 2 are kept.")
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
    st.caption("Use: place B next to A in the app, bundle them, or suggest B at checkout when A is in the cart.")

# ------------------------------------------------------------------------------------------- recommendations
with recs:
    tab_intro("What should we show each customer next, and does it work?")
    m = q.recommendation_metrics(role).set_index("method")
    st.markdown("**How good are the recommendations?** Weights chosen on August, model trained on April–August, "
                "checked once against what each customer actually bought in September.")
    c = st.columns(3)
    for col, method, title in zip(c, ("hybrid", "repeat", "popularity"),
                                  ("Hybrid (used)", "Buy-again only", "Most popular")):
        if method in m.index:
            col.metric(title, f"{m.loc[method, 'precision_at_10']:.1%}",
                       help=f"Precision@10; hit rate {m.loc[method, 'hit_rate']:.0%}, "
                            f"recall@10 {m.loc[method, 'recall_at_10']:.1%}, coverage {m.loc[method, 'coverage']:.0%}")
    if "hybrid" in m.index:
        st.caption(f"Hybrid weights: {m.loc['hybrid', 'weights']}.")
    st.caption("Precision@10 = share of the 10 recommended products the customer bought. In this data only about a "
               "quarter of purchases are repeats of an item the customer bought before, so best-sellers are a strong "
               "baseline; the hybrid mixes buy-again, 'often bought with' and best-sellers. Full table below.")
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
with retention:
    tab_intro("Who is still ordering, and who is worth winning back?")
    st.info("Descriptive only: who is still ordering and who has gone quiet. No churn **prediction** is made — the "
            "data has no reliable churn signal to train on (a predictive model is a later, Tier 2 item).", icon="ℹ️")
    st.markdown("**Cohort retention** — customers grouped by the month of their first completed order; each cell is "
                "the share who ordered again that many months later.")
    st.plotly_chart(cohort_heatmap(q.retention_cohorts(role)), width="stretch")
    status = q.retention_status(role)
    st.markdown("**Status at the end of the data** — active ≤ 14 days since last order, cooling 15–30, at risk 31–60, "
                "lapsed > 60")
    grid = status.pivot_table(index="status", columns="value_tier", values="customers", fill_value=0, sort=False)
    st.dataframe(grid, width="stretch")
    f1, f2, f3 = st.columns([2, 2, 1])
    sts = f1.multiselect("Status", ["active", "cooling", "at_risk", "lapsed"], default=["at_risk", "lapsed"])
    tiers = f2.multiselect("Value tier", ["High", "Medium", "Low"], default=["High"])
    overdue = f3.checkbox("Overdue only", help="Days since last order > 2 × their own average gap")
    who = q.retention_customers(role, sts or None, tiers or None, overdue)
    st.caption(f"{len(who)} customers (highest spend first) — a win-back list.")
    st.dataframe(who, hide_index=True, width="stretch",
                 column_config={"total_spend": st.column_config.NumberColumn("spend (₹)", format="%.0f")})

# ------------------------------------------------------------------------------------------- promotions
with promos:
    tab_intro("Did our promotions lift sales, and what did they cost?")
    pm = q.promotion_metrics(role)
    st.caption("Uplift = average daily units of the promoted category (all stores) during the promotion vs days when "
               "that category had no promotion. It is a simple before/after comparison, not a controlled experiment: "
               "seasonality and other events can also move sales.")
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
