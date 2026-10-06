"""Marketing analytics (Phase 5D): segmentation, basket rules, recommendations, retention — one stage step.

- segmentation    : src/ml/segmentation.py (K-means; named from measured profiles)
- basket rules    : directed rules A → B from gld_basket_pairs (confidence ≥ 10%, lift ≥ 2)
- recommendations : src/ml/recommendations.py (buy again + often bought with + popularity; weights chosen on
                    August, September backtest)
- retention       : descriptive cohorts and per-customer status (the synthetic data has no planted churn, so no
                    churn model is trained — see Project_Plan_v2.md §10.1)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.io import read_csv_strings
from src.ml import recommendations, segmentation
from src.ml.tables import (
    BASKET_RULES,
    CUSTOMER_RETENTION,
    CUSTOMER_SEGMENTS,
    RECOMMENDATION_METRICS,
    RECOMMENDATIONS,
    RETENTION_COHORTS,
    SEGMENT_PROFILES,
    SEGMENTATION_SELECTION,
    read_table,
    write_table,
)
from src.orchestration.tracking import RunTracker, utc_now
from src.transformations.gold.customers import BASKET_PAIRS, CUSTOMER_360, CUSTOMER_CATEGORY

MIN_CONFIDENCE, MIN_LIFT = 0.10, 2.0
STATUS_BANDS = [(14, "active"), (30, "cooling"), (60, "at_risk")]


def _silver(silver_root: Path, dataset: str) -> pd.DataFrame:
    return pd.concat([read_csv_strings(p) for p in sorted((silver_root / f"slv_{dataset}").glob("*.csv"))],
                     ignore_index=True)


def basket_rules(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b, na, nb, ca, cb, conf in (("product_a", "product_b", "product_a_name", "product_b_name", "category_a",
                                        "category_b", "confidence_a_to_b"),
                                       ("product_b", "product_a", "product_b_name", "product_a_name", "category_b",
                                        "category_a", "confidence_b_to_a")):
        rows.append(pd.DataFrame({
            "antecedent_id": pairs[a], "antecedent": pairs[na], "antecedent_category": pairs[ca],
            "consequent_id": pairs[b], "consequent": pairs[nb], "consequent_category": pairs[cb],
            "baskets_both": pairs["baskets_both"], "support": pairs["support"], "confidence": pairs[conf],
            "lift": pairs["lift"]}))
    rules = pd.concat(rows, ignore_index=True)
    return rules[(rules["confidence"] >= MIN_CONFIDENCE) & (rules["lift"] >= MIN_LIFT)].sort_values(
        "lift", ascending=False, kind="stable")


def retention(orders: pd.DataFrame, customers: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    done = orders[orders["is_completed"] == "true"].copy()
    done["day"] = pd.to_datetime(done["business_date"])
    done["value"] = pd.to_numeric(done["total_amount"])
    done["month"] = done["day"].dt.to_period("M")
    first = done.groupby("customer_id")["month"].min().rename("cohort")
    active = done[["customer_id", "month"]].drop_duplicates().join(first, on="customer_id")
    active["offset"] = (active["month"] - active["cohort"]).apply(lambda x: x.n)
    size = first.value_counts().rename("cohort_size")
    cohorts = active.groupby(["cohort", "offset"]).size().rename("active_customers").reset_index().join(size, on="cohort")
    cohorts["retention_rate"] = (cohorts["active_customers"] / cohorts["cohort_size"]).round(4)
    cohorts = cohorts.rename(columns={"cohort": "cohort_month", "offset": "month_offset"})
    cohorts["cohort_month"] = cohorts["cohort_month"].dt.to_timestamp()

    per = done.groupby("customer_id").agg(first_order_date=("day", "min"), last_order_date=("day", "max"),
                                          orders_completed=("order_id", "nunique"), total_spend=("value", "sum"))
    per["days_since_last"] = (as_of - per["last_order_date"]).dt.days
    span = (per["last_order_date"] - per["first_order_date"]).dt.days
    per["avg_days_between"] = np.where(per["orders_completed"] > 1, (span / (per["orders_completed"] - 1)).round(1), np.nan)
    per["status"] = "lapsed"
    for limit, name in reversed(STATUS_BANDS):
        per.loc[per["days_since_last"] <= limit, "status"] = name
    per["value_tier"] = pd.qcut(per["total_spend"].rank(method="first"), 3, labels=["Low", "Medium", "High"]).astype(str)
    per["overdue"] = per["days_since_last"] > 2 * per["avg_days_between"]
    per = per.reset_index()
    never = customers[~customers["customer_id"].isin(per["customer_id"])][["customer_id"]].assign(
        status="never_ordered", value_tier="None", orders_completed=0, total_spend=0.0, overdue=False)
    return cohorts, pd.concat([per, never], ignore_index=True)


def run_marketing(cfg: Config, tracker: RunTracker, gold_root: Path, silver_root: Path) -> dict:
    t0, started = time.perf_counter(), utc_now()
    version = f"marketing_{tracker.run_id}"
    c360, cat = read_table(gold_root, CUSTOMER_360), read_table(gold_root, CUSTOMER_CATEGORY)
    categories = _silver(silver_root, "categories")
    feats = segmentation.build_features(c360, cat, dict(zip(categories["category_id"], categories["category_name"])))
    seg = segmentation.segment(feats, cfg.seed)
    rules = basket_rules(read_table(gold_root, BASKET_PAIRS))
    orders, items = _silver(silver_root, "orders"), _silver(silver_root, "order_items")
    products = _silver(silver_root, "products")
    recs = recommendations.backtest_and_fit(orders, items, cfg.ml_split.train_end, len(products))
    cohorts, status = retention(orders, _silver(silver_root, "customers"), pd.Timestamp(cfg.calendar.end_date))

    outputs = {
        CUSTOMER_SEGMENTS: seg["assignments"], SEGMENT_PROFILES: seg["profiles"], SEGMENTATION_SELECTION: seg["selection"],
        BASKET_RULES: rules, RECOMMENDATIONS: recs["recommendations"], RECOMMENDATION_METRICS: recs["metrics"],
        RETENTION_COHORTS: cohorts, CUSTOMER_RETENTION: status,
    }
    counts = {}
    for table, frame in outputs.items():
        frame = frame.copy()
        if "model_version" in table.column_names:
            frame["model_version"] = version
        counts[table.name] = write_table(frame, gold_root, table)
        tracker.table(stage="ml", job=table.name, source=",".join(table.sources), target=table.name, engine="python",
                      module="src.ml.marketing", rows_written=counts[table.name], status="success", started_at=started,
                      ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        for source in table.sources:
            tracker.lineage(source, table.name, table.name, engine="python")

    rec_m = recs["metrics"].set_index("method")
    tracker.model_run(model_name="customer_segmentation", model_version=f"{version}/segmentation",
                      method=f"KMeans k={seg['k']} (chosen by silhouette from 3–8), StandardScaler",
                      trained_at=utc_now(), input_table="gld_customer_360|gld_customer_category",
                      train_rows=len(feats), features="|".join(feats.columns),
                      params=json.dumps({"k": seg["k"], "seed": cfg.seed}),
                      metrics=json.dumps({"silhouette": seg["silhouette"], "stability_ari": round(seg["stability"], 4)}),
                      status="current", limitations="Segments are named from measured traits; synthetic data.")
    tracker.model_run(model_name="recommendations", model_version=f"{version}/recommendations",
                      method=f"hybrid: {recs['metrics'].set_index('method').loc['hybrid', 'weights']} "
                             "(chosen on August from a fixed grid)",
                      trained_at=utc_now(), train_start=cfg.calendar.start_date.isoformat(),
                      train_end=cfg.ml_split.train_end.isoformat(), test_start=cfg.ml_split.test_start.isoformat(),
                      test_end=cfg.calendar.end_date.isoformat(), input_table="slv_orders|slv_order_items",
                      params=json.dumps({"weights": recs["weights"], "grid": recommendations.WEIGHT_GRID}),
                      metrics=json.dumps(rec_m[["precision_at_10", "recall_at_10", "hit_rate", "coverage"]].to_dict("index")),
                      status="current", limitations="Weights revised after the first measurement (fixed weights lost to "
                      "popularity); chosen on August, tested once on September; final model refit on all data.")
    for method, row in rec_m.iterrows():
        for metric in ("precision_at_10", "recall_at_10", "hit_rate", "coverage"):
            tracker.model_metric(model_version=f"{version}/recommendations", evaluation="recommendation_backtest",
                                 model=method, horizon=0, segment="all", metric=metric, value=row[metric])
    for g in recs["grid"].itertuples():
        tracker.model_metric(model_version=f"{version}/recommendations", evaluation="weight_selection_august",
                             model=f"repeat={g.repeat:g}|copurchase={g.copurchase:g}|popular={g.popular:g}", horizon=0,
                             segment="all", metric="precision_at_10", value=g.precision_at_10)
    return {"segments": seg["k"], "silhouette": round(seg["silhouette"], 3), "stability": round(seg["stability"], 3),
            "rules": len(rules), "precision_at_10": rec_m["precision_at_10"].to_dict(), "weights": recs["weights"],
            "status": status["status"].value_counts().to_dict()}
