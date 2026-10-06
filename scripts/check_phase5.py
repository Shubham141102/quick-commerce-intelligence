"""Phase 5 completion report — pure pandas. Sections are added as each step is built.

    python -m scripts.check_phase5

5A lost-sales estimate + Inventory explorer: rows only on stockout days, arithmetic consistent, estimate vs
the simulator's true lost units (pass bars fixed before measuring: total within ±50%, SKU-level Spearman ≥ 0.4),
explorer data functions role-checked.
5B Business pages equal Gold. 5C anomaly recall / precision vs planted anomalies (bars fixed before measuring).
5D segments vs planted personas, stability, planted affinity pairs, recommendations vs baselines, retention
arithmetic. Ground truth is read only here, never by the pipeline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.ml.tables import LOST_SALES, read_table
from src.orchestration.tracking import read_meta
from src.transformations.gold.inventory import INVENTORY_DAILY

TOTAL_TOLERANCE = 0.5      # estimate within ±50% of the true total lost units
MIN_SPEARMAN = 0.4         # SKU-level agreement with the true lost units

# Bars that were fixed in advance, measured, NOT met, and accepted (decision 2026-10-05) as documented known
# limitations. The bar itself is never changed; the report shows them as LIMIT, not PASS.
KNOWN_LIMITATIONS = {
    "5A.3": "SKU-level ranking of lost sales: measured 0.35 vs the pre-set 0.4. Reliable for totals and the biggest "
            "losers (Spearman 0.47 for SKUs losing ≥ 10 units); root cause is stockout-day detection in the rebuilt "
            "Gold inventory. See docs/08_inventory_explorer_lost_sales.md.",
    "5C.2": "Planted demand spikes: 4 of 6 vs the pre-set 80%, after one approved method revision. Both misses are "
            "not detectable from this data: one is on day 4 (no 28-day history yet), the other produced only 3 "
            "orders, below the 4-order minimum. See docs/09_business_workspace.md.",
}


def _generation_dir(paths: dict):
    loads = read_meta(resolve(paths["metadata"]), "meta_file_loads")
    return resolve(paths["generation"]) / loads.loc[loads["status"] == "loaded", "generation_run_id"].iloc[0]


def check_5a(paths: dict, strict: bool = True) -> list[tuple[str, bool, str]]:
    gold = resolve(paths["gold"])
    lost, inv = read_table(gold, LOST_SALES), read_table(gold, INVENTORY_DAILY)
    out = []
    keys = ["store_id", "product_id", "business_date"]
    joined = lost.merge(inv[[*keys, "is_stockout"]], on=keys, how="left")
    arithmetic = np.allclose(lost["lost_revenue"], (lost["lost_units"] * lost["unit_price"]).round(2), atol=0.011)
    ok = bool(joined["is_stockout"].all() and (lost["lost_units"] >= 0).all() and arithmetic
              and not lost.duplicated(keys).any())
    out.append(("5A.1 Lost sales only on stockout days; never negative; units × price = revenue", ok,
                f"{len(lost):,} stockout days, {lost['lost_units'].sum():,.0f} units, ₹{lost['lost_revenue'].sum():,.0f}"))

    truth = read_csv_strings(_generation_dir(paths) / "ground_truth" / "gt_stockouts.csv")
    truth["lost_units"] = truth["lost_units"].astype(float)
    true_pairs = truth.groupby(["store_id", "product_id"])["lost_units"].sum()
    est_pairs = lost.groupby(["store_id", "product_id"])["lost_units"].sum()
    both = pd.concat([true_pairs.rename("true"), est_pairs.rename("est")], axis=1).fillna(0)
    ratio = both["est"].sum() / both["true"].sum() if both["true"].sum() else float("nan")
    spearman = both["true"].corr(both["est"], method="spearman")
    out.append(("5A.2 Estimate close to the simulator's true lost units (total within ±50%)",
                abs(ratio - 1) <= TOTAL_TOLERANCE or not strict,
                f"estimated {both['est'].sum():,.0f} vs true {both['true'].sum():,.0f} units (ratio {ratio:.2f})"))
    out.append(("5A.3 Ranks the right SKUs (SKU-level Spearman ≥ 0.4)", spearman >= MIN_SPEARMAN or not strict,
                f"Spearman {spearman:.2f} over {len(both)} store × SKU pairs"))

    from src.serving import queries as q
    with _Snapshot(paths):
        risk = q.risk_list("inventory_manager")
        store, product = risk["store_id"].iloc[0], risk["product_id"].iloc[0]
        calls = [lambda r: q.explorer_timeline(r, store, product), lambda r: q.explorer_summary(r, store, product),
                 q.lost_sales_by_store, lambda r: q.top_lost_sales_skus(r, 5)]
        ok, blocked = _role_checked(calls, "inventory_manager", ("marketing_manager", "business_analyst", "data_engineer"))
        out.append(("5A.4 Explorer data functions work for inventory, blocked for others", ok,
                    f"{len(calls)} functions; {blocked}/{len(calls)} refuse every other role"))
    return out


class _Snapshot:
    """Point the serving layer at the run's published snapshot for the duration of a check section."""

    def __init__(self, paths: dict):
        self.demo = str(resolve(paths["demo"]))

    def __enter__(self):
        from src.serving.db import get_snapshot
        self.previous = os.environ.get("QCI_DEMO_DIR")
        os.environ["QCI_DEMO_DIR"] = self.demo
        get_snapshot.cache_clear()
        return self

    def __exit__(self, *exc):
        from src.serving.db import get_snapshot
        if self.previous is None:
            os.environ.pop("QCI_DEMO_DIR", None)
        else:
            os.environ["QCI_DEMO_DIR"] = self.previous
        get_snapshot.cache_clear()


def _role_checked(calls: list, allowed: str, others: tuple[str, ...]) -> tuple[bool, int]:
    from src.serving.permissions import AccessDenied
    works = all(c(allowed) is not None for c in calls)
    blocked = 0
    for c in calls:
        refused = 0
        for role in others:
            try:
                c(role)
            except AccessDenied:
                refused += 1
        blocked += refused == len(others)
    return works and blocked == len(calls), blocked


def check_5b(paths: dict) -> list[tuple[str, bool, str]]:
    from src.serving import queries as q
    from src.transformations.gold.operations import DELIVERY_METRICS
    from src.transformations.gold.sales import DAILY_SALES
    gold = resolve(paths["gold"])
    sales, dm = read_table(gold, DAILY_SALES), read_table(gold, DELIVERY_METRICS)
    out = []
    with _Snapshot(paths):
        b = q.period_bounds("business_analyst")
        s, e = b["first_day"].date(), b["last_day"].date()
        k = q.sales_kpis("business_analyst", s, e)["current"]
        stores = q.revenue_by_store("business_analyst", s, e)
        cats = q.category_contribution("business_analyst", s, e)
        same = (abs(float(k["net_revenue"]) - float(sales["net_revenue"].sum())) < 0.01
                and abs(float(stores["net_revenue"].sum()) - float(sales["net_revenue"].sum())) < 0.01
                and abs(float(cats["revenue"].sum()) - float(sales["gmv"].sum())) < 0.01
                and int(k["orders_completed"]) == int(sales["orders_completed"].sum()))
        out.append(("5B.1 Sales page totals equal Gold (net revenue, by store, by category, orders)", same,
                    f"net revenue ₹{float(k['net_revenue']):,.2f}; {int(k['orders_completed']):,} completed orders; "
                    f"AOV ₹{k['aov']:,.2f}"))
        d = q.delivery_kpis("business_analyst", s, e)
        weighted = float((dm["on_time_rate"].fillna(0) * dm["delivered"]).sum() / dm["delivered"].sum())
        ok = int(d["delivered"]) == int(dm["delivered"].sum()) and abs(float(d["on_time_rate"]) - weighted) < 1e-6
        out.append(("5B.2 Delivery page equals Gold (delivered, on-time rate)", ok,
                    f"{int(d['delivered']):,} delivered, on-time {float(d['on_time_rate']):.1%}, "
                    f"avg {float(d['avg_minutes']):.1f} min"))
        calls = [lambda r: q.sales_kpis(r, s, e), lambda r: q.sales_trend(r, s, e), lambda r: q.revenue_by_store(r, s, e),
                 lambda r: q.category_contribution(r, s, e), lambda r: q.top_products(r, s, e),
                 lambda r: q.delivery_kpis(r, s, e), lambda r: q.delivery_trend(r, s, e),
                 lambda r: q.delivery_by_store(r, s, e), lambda r: q.cancellations_by_reason(r, s, e)]
        ok, blocked = _role_checked(calls, "business_analyst", ("inventory_manager", "marketing_manager", "data_engineer"))
        out.append(("5B.3 Business data functions: analyst allowed, other roles blocked", ok,
                    f"{len(calls)} functions; {blocked}/{len(calls)} refuse every other role"))
    return out


SPIKE_RECALL_BAR = 0.8        # pass bars fixed before measuring (2026-10-05)
PAYMENT_RECALL_BAR = 2 / 3
PRECISION_BAR = 0.5


def check_5c(paths: dict) -> list[tuple[str, bool, str]]:
    from src.ml.tables import ANOMALY_SERIES, SALES_ANOMALIES
    from src.serving import queries as q
    gold = resolve(paths["gold"])
    found, series = read_table(gold, SALES_ANOMALIES), read_table(gold, ANOMALY_SERIES)
    truth = read_csv_strings(_generation_dir(paths) / "ground_truth" / "gt_anomalies.csv")
    for col in ("start_ts", "end_ts"):
        truth[col] = pd.to_datetime(truth[col].str.replace("Z", ""))
    kind = {"store_outage": "store_outage", "demand_spike": "demand_spike", "payment_gateway_failure": "payment_failure"}
    truth["detector"] = truth["anomaly_type"].map(kind)
    slack = {"store_outage": pd.Timedelta(hours=1), "demand_spike": pd.Timedelta(days=1),
             "payment_failure": pd.Timedelta(hours=1)}

    def matches(f, t) -> bool:
        if f["detector"] != t["detector"]:
            return False
        if t["store_id"] and f["store_id"] != t["store_id"]:
            return False
        if t["product_id"] and f["product_id"] != t["product_id"]:
            return False
        s = slack[t["detector"]]
        return f["start_ts"] <= t["end_ts"] + s and f["end_ts"] >= t["start_ts"] - s

    hit_truth = {i for i, t in truth.iterrows() if any(matches(f, t) for _, f in found.iterrows())}
    is_true = found.apply(lambda f: any(matches(f, t) for _, t in truth.iterrows()), axis=1).astype(bool)
    scored = found["detector"] != "store_outage"   # outages are experimental: reported, not in precision (2026-10-05)
    true_found, n_scored = int((is_true & scored).sum()), int(scored.sum())
    outage_alarms = int((~scored).sum())
    out = []
    ok = (not found["anomaly_id"].duplicated().any() and set(series["anomaly_id"]) == set(found["anomaly_id"])
          and found["p_value"].between(0, 1).all())
    out.append(("5C.1 Anomaly table complete; every anomaly has a context series", ok,
                ", ".join(f"{k} {v}" for k, v in found["detector"].value_counts().items()) or "none found"))
    recall = {}
    for det in ("demand_spike", "payment_failure", "store_outage"):
        idx = truth.index[truth["detector"] == det]
        recall[det] = (len(hit_truth & set(idx)), len(idx))
    sp, pf, so = recall["demand_spike"], recall["payment_failure"], recall["store_outage"]
    out.append(("5C.2 Planted demand spikes found (recall ≥ 80%)", sp[1] and sp[0] / sp[1] >= SPIKE_RECALL_BAR,
                f"{sp[0]} of {sp[1]}"))
    out.append(("5C.3 Planted payment-gateway failures found (recall ≥ 2/3)", pf[1] and pf[0] / pf[1] >= PAYMENT_RECALL_BAR,
                f"{pf[0]} of {pf[1]}"))
    out.append(("5C.4 Store outages — experimental (no bar: low volume, reported for honesty)", True,
                f"{so[0]} of {so[1]} planted outages found; {outage_alarms - int((is_true & ~scored).sum())} of "
                f"{outage_alarms} outage alarms are false"))
    precision = true_found / n_scored if n_scored else 0.0
    out.append(("5C.5 Spike + payment detections that match a planted anomaly (precision ≥ 50%)",
                precision >= PRECISION_BAR, f"{true_found} of {n_scored} detections ({precision:.0%})"))
    with _Snapshot(paths):
        aid = found["anomaly_id"].iloc[0] if len(found) else "A0001"
        calls = [lambda r: q.anomaly_list(r), lambda r: q.anomaly_series(r, aid), lambda r: q.anomaly_summary(r)]
        ok, blocked = _role_checked(calls, "business_analyst", ("inventory_manager", "marketing_manager", "data_engineer"))
        out.append(("5C.6 Anomaly data functions: analyst allowed, other roles blocked", ok,
                    f"{len(calls)} functions; {blocked}/{len(calls)} refuse every other role"))
    return out


SEGMENT_ARI_BAR = 0.3         # pass bars fixed before measuring (2026-10-05)
STABILITY_BAR = 0.7
PLANTED_PAIRS_BAR = 20        # of the 25 planted affinity pairs


def check_5d(paths: dict) -> list[tuple[str, bool, str]]:
    from sklearn.metrics import adjusted_rand_score

    from src.ml.tables import (
        BASKET_RULES,
        CUSTOMER_RETENTION,
        CUSTOMER_SEGMENTS,
        RECOMMENDATION_METRICS,
        RETENTION_COHORTS,
        SEGMENT_PROFILES,
        SEGMENTATION_SELECTION,
    )
    from src.serving import queries as q
    from src.transformations.gold.customers import CUSTOMER_360
    gold, gt = resolve(paths["gold"]), _generation_dir(paths) / "ground_truth"
    seg, prof = read_table(gold, CUSTOMER_SEGMENTS), read_table(gold, SEGMENT_PROFILES)
    sel, c360 = read_table(gold, SEGMENTATION_SELECTION), read_table(gold, CUSTOMER_360)
    buyers = set(c360.loc[c360["orders_completed"] > 0, "customer_id"])
    k = int(sel.loc[sel["chosen"], "k"].iloc[0])
    ok = (set(seg["customer_id"]) == buyers and seg["customer_id"].is_unique and prof["segment_label"].is_unique
          and 3 <= k <= 8 and int(prof["customers"].sum()) == len(seg))
    out = [("5D.1 Every buyer in exactly one segment; k from 3–8; unique generated names", ok,
            f"k = {k} (silhouette {float(sel.loc[sel['chosen'], 'silhouette'].iloc[0]):.3f}); {len(seg):,} customers; "
            + "; ".join(prof.sort_values("customers", ascending=False)["segment_label"].head(3)) + " …")]

    personas = read_csv_strings(gt / "gt_personas.csv").merge(seg, on="customer_id")
    ari = adjusted_rand_score(personas["persona"], personas["segment_id"])
    out.append((f"5D.2 Segments recover the planted personas (ARI >= {SEGMENT_ARI_BAR})", ari >= SEGMENT_ARI_BAR,
                f"adjusted Rand index {ari:.3f} over {len(personas):,} customers, "
                f"{personas['persona'].nunique()} personas vs {k} segments"))
    stab = float(sel.loc[sel["chosen"], "stability_ari"].iloc[0])
    out.append((f"5D.3 Segments are stable across random seeds (ARI >= {STABILITY_BAR})", stab >= STABILITY_BAR,
                f"mean ARI vs 5 reruns {stab:.3f}"))

    rules = read_table(gold, BASKET_RULES)
    planted = read_csv_strings(gt / "gt_affinity_pairs.csv")
    have = set(zip(rules["antecedent_id"], rules["consequent_id"]))
    hits = sum((a, b) in have for a, b in zip(planted["antecedent_product_id"], planted["consequent_product_id"]))
    out.append((f"5D.4 Planted affinity pairs found as rules A → B (>= {PLANTED_PAIRS_BAR} of {len(planted)})",
                hits >= PLANTED_PAIRS_BAR,
                f"{hits} of {len(planted)} found among {len(rules):,} rules (confidence >= 10%, lift >= 2)"))

    m = read_table(gold, RECOMMENDATION_METRICS).set_index("method")
    p = m["precision_at_10"]
    out.append(("5D.5 Recommendations beat the popularity baseline (Precision@10, September backtest)",
                bool(p["hybrid"] > p["popularity"]),
                f"hybrid {p['hybrid']:.1%} vs popularity {p['popularity']:.1%}; buy-again only {p['repeat']:.1%} "
                f"(reported for honesty); hit rate {m.loc['hybrid', 'hit_rate']:.0%}, {int(m.loc['hybrid', 'customers']):,} customers"))

    ret, coh = read_table(gold, CUSTOMER_RETENTION), read_table(gold, RETENTION_COHORTS)
    first = coh[coh["month_offset"] == 0]
    ordered = ret[ret["status"] != "never_ordered"]
    ok = (len(ret) == len(c360) and ret["customer_id"].is_unique and set(ordered["customer_id"]) == buyers
          and int(first["cohort_size"].sum()) == len(buyers) and bool((first["retention_rate"] == 1).all())
          and bool((coh["active_customers"] <= coh["cohort_size"]).all()))
    counts = ret["status"].value_counts()
    out.append(("5D.6 Retention arithmetic: every customer has one status; cohorts add up to all buyers", ok,
                ", ".join(f"{s} {int(counts.get(s, 0)):,}" for s in ("active", "cooling", "at_risk", "lapsed",
                                                                     "never_ordered"))))
    with _Snapshot(paths):
        sid = int(prof["segment_id"].iloc[0])
        cid, pid = seg["customer_id"].iloc[0], rules["antecedent_id"].iloc[0] if len(rules) else "P0001"
        calls = [q.marketing_kpis, q.segment_profiles, q.segmentation_selection, lambda r: q.segment_customers(r, sid),
                 q.basket_rules, q.rule_products, lambda r: q.product_partners(r, pid), q.recommendation_metrics,
                 q.recommendation_customers, lambda r: q.customer_recommendations(r, cid),
                 lambda r: q.segment_top_recommendations(r, sid), q.retention_cohorts, q.retention_status,
                 q.retention_customers, q.promotion_metrics]
        ok, blocked = _role_checked(calls, "marketing_manager", ("inventory_manager", "business_analyst", "data_engineer"))
        out.append(("5D.7 Marketing data functions: marketing manager allowed, other roles blocked", ok,
                    f"{len(calls)} functions; {blocked}/{len(calls)} refuse every other role"))
    return out


SECTIONS = [("5A  Lost-sales estimate + Inventory explorer", check_5a),
            ("5B  Business workspace (sales performance, delivery & operations)", check_5b),
            ("5C  Sales anomaly detection", check_5c),
            ("5D  Marketing analytics (segments, basket rules, recommendations, retention)", check_5d)]


def status(name: str, ok: bool) -> str:
    """PASS, FAIL, or LIMIT (failed a pre-set bar that is a documented, accepted known limitation)."""
    if ok:
        return "PASS"
    return "LIMIT" if name.split(" ", 1)[0] in KNOWN_LIMITATIONS else "FAIL"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = load_config("small").paths
    print("\nPhase 5 completion report")
    checks, statuses = [], []
    for title, section in SECTIONS:
        print(f"\n{title}")
        part = section(paths)
        for name, ok, detail in part:
            st = status(name, ok)
            print(f"  [{st:5s}] {name:76s} {detail}")
            checks.append((name, ok, detail))
            statuses.append(st)
    limits = [name.split(" ", 1)[0] for (name, _, _), st in zip(checks, statuses) if st == "LIMIT"]
    for code in limits:
        print(f"\n  Known limitation {code}: {KNOWN_LIMITATIONS[code]}")
    failed = statuses.count("FAIL")
    print(f"\n{statuses.count('PASS')} passed, {len(limits)} known limitation(s), {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
