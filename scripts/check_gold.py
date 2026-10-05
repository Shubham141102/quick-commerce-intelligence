"""Gold completion report (Project_Plan_v2.md §7.10) — pure pandas, recomputed independently from Silver.

    python -m scripts.check_gold

Core checks (must pass): tables written, grain unique, complete grids, revenue/discount/refunds/orders
reconcile with Silver to the paisa, inventory continuity, no leakage in forecasting features.
Signal checks (planted patterns visible): weekend and rain uplift, planted product pairs, planted stockouts.
Ground truth is read only here, never by the pipeline.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.orchestration.tracking import read_meta

GOLD_GRAINS = {
    "gld_daily_sales": ["store_id", "business_date"],
    "gld_daily_category_sales": ["store_id", "category_id", "business_date"],
    "gld_product_performance": ["product_id", "month"],
    "gld_inventory_daily": ["store_id", "product_id", "business_date"],
    "gld_demand_features": ["store_id", "category_id", "business_date"],
    "gld_customer_360": ["customer_id"],
    "gld_customer_category": ["customer_id", "category_id"],
    "gld_basket_pairs": ["product_a", "product_b"],
    "gld_delivery_metrics": ["store_id", "business_date"],
    "gld_cancellation_metrics": ["store_id", "week_start", "stage", "reason"],
    "gld_promotion_metrics": ["promotion_id"],
    "gld_quality_summary": ["silver_run_id", "dataset", "check_type", "rule", "column"],
}


def _read(folder: Path) -> pd.DataFrame:
    parts = sorted(folder.glob("*.csv"))
    return pd.concat([read_csv_strings(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def _money(series: pd.Series) -> Decimal:
    return sum((Decimal(v) for v in series if v != ""), Decimal("0"))


def evaluate(paths: dict, profile: str) -> tuple[list[tuple[str, bool, str]], list[tuple[str, bool, str]], dict]:
    cfg = load_config(profile)
    silver, gold = resolve(paths["silver"]), resolve(paths["gold"])
    s = {n: _read(silver / f"slv_{n}") for n in ["orders", "order_items", "products", "order_promotions",
                                                  "returns_refunds", "stores", "categories", "customers",
                                                  "inventory_snapshots"]}
    g = {n: _read(gold / n) for n in GOLD_GRAINS}
    core, signals, info = [], [], {}

    # 1-3. tables, grain, grids
    empty = [n for n, df in g.items() if df.empty]
    core.append(("1. Every Gold table written", not empty, "12/12" if not empty else f"empty: {empty}"))
    dups = {n: int(df.duplicated(GOLD_GRAINS[n]).sum()) for n, df in g.items() if not df.empty}
    core.append(("2. Grain unique in every table", not any(dups.values()), f"duplicates: {sum(dups.values())}"))
    days, stores, cats = cfg.calendar.n_days, len(s["stores"]), len(s["categories"])
    pairs = s["inventory_snapshots"][["store_id", "product_id"]].drop_duplicates().shape[0]
    grids = {"gld_daily_sales": stores * days, "gld_daily_category_sales": stores * cats * days,
             "gld_inventory_daily": pairs * days, "gld_demand_features": stores * cats * days,
             "gld_customer_360": len(s["customers"]), "gld_delivery_metrics": stores * days}
    bad = {n: (len(g[n]), want) for n, want in grids.items() if len(g[n]) != want}
    core.append(("3. Complete grids (zero-filled days included)", not bad,
                 f"{len(g['gld_daily_sales']):,} store-days, {len(g['gld_inventory_daily']):,} SKU-days" if not bad else str(bad)))

    # 4. reconciliation with Silver, recomputed independently
    o = s["orders"]
    completed = set(o.loc[o["is_completed"] == "true", "order_id"])
    items = s["order_items"].merge(s["products"][["product_id", "price"]], on="product_id")
    items = items[items["order_id"].isin(completed)]
    flagged = items["dq_unit_price_mismatch"] == "true"
    revenue = (items["line_amount"].map(Decimal) * ~flagged).sum() + sum(
        (Decimal(q) * Decimal(p)).quantize(Decimal("0.01")) for q, p in zip(items.loc[flagged, "quantity"], items.loc[flagged, "price"]))
    disc = _money(s["order_promotions"].loc[s["order_promotions"]["order_id"].isin(completed), "discount_amount"])
    rf = s["returns_refunds"]
    refunds = _money(rf.loc[rf["status"] == "completed", "amount"])
    ds, dc, pp = g["gld_daily_sales"], g["gld_daily_category_sales"], g["gld_product_performance"]
    checks = {
        "GMV (daily sales)": (_money(ds["gmv"]), revenue),
        "revenue (category sales)": (_money(dc["revenue"]), revenue),
        "revenue (product performance)": (_money(pp["revenue"]), revenue),
        "discount": (_money(ds["discount"]), disc),
        "refunds": (_money(ds["refunds"]), refunds),
        "completed orders": (Decimal(int(ds["orders_completed"].astype(int).sum())), Decimal(len(completed))),
        "orders placed": (Decimal(int(ds["orders_placed"].astype(int).sum())), Decimal(len(o))),
        "units (daily vs category)": (Decimal(int(ds["units"].astype(int).sum())), Decimal(int(dc["units"].astype(int).sum()))),
    }
    off = {k: (str(a), str(b)) for k, (a, b) in checks.items() if a != b}
    net = _money(ds["net_revenue"])
    info["net_revenue"], info["gmv"], info["completed_orders"] = net, revenue, len(completed)
    core.append(("4. Revenue, discount, refunds, orders reconcile with Silver", not off and net == revenue - disc - refunds,
                 f"GMV ₹{revenue:,} − discount ₹{disc:,} − refunds ₹{refunds:,} = net ₹{net:,}" if not off else str(off)))

    # 5. inventory continuity + reconciliation gap
    inv = g["gld_inventory_daily"].sort_values(["store_id", "product_id", "business_date"])
    nxt = inv.groupby(["store_id", "product_id"])["opening_stock"].shift(-1)
    breaks = int(((nxt.notna()) & (nxt != inv["closing_stock"])).sum())
    snap_days = int((inv["snapshot_stock"] != "").sum())
    gaps = inv.loc[inv["reconciliation_gap"] != "", "reconciliation_gap"].astype(int)
    info["recon_gap_mean_abs"] = float(gaps.abs().mean()) if len(gaps) else 0.0
    info["recon_gap_zero_share"] = float((gaps == 0).mean()) if len(gaps) else 1.0
    info["calculated_negative_days"] = int((inv["calculated_negative"] == "true").sum())
    core.append(("5. Inventory: closing stock = next day's opening; every count anchored",
                 breaks == 0 and snap_days == len(s["inventory_snapshots"]),
                 f"{breaks} breaks; {snap_days:,} counted days; reconciliation gap 0 at "
                 f"{info['recon_gap_zero_share']:.0%} of counts (mean |gap| {info['recon_gap_mean_abs']:.2f} units)"))

    # 6. leakage: recompute lags / rolling means from the daily series using only earlier days
    f = g["gld_demand_features"].merge(dc[["store_id", "category_id", "business_date", "units"]],
                                       on=["store_id", "category_id", "business_date"])
    f = f.sort_values(["store_id", "category_id", "business_date"])
    f["units"] = f["units"].astype(int)
    grp = f.groupby(["store_id", "category_id"])["units"]
    expect = {"lag_1": grp.shift(1), "lag_7": grp.shift(7),
              "rolling_mean_7": grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean().round(3))}
    mismatch = {}
    for col, exp in expect.items():
        got = pd.to_numeric(f[col].replace("", np.nan))
        mismatch[col] = int((~np.isclose(got.fillna(-1), exp.fillna(-1), atol=1e-3)).sum())
    core.append(("6. Forecast features use only earlier days (no leakage)", not any(mismatch.values()),
                 "lag_1, lag_7, rolling_mean_7 match a past-only recomputation" if not any(mismatch.values()) else str(mismatch)))

    # 7. signals
    ds["units_i"] = ds["units"].astype(int)
    ds["dow"] = pd.to_datetime(ds["business_date"]).dt.dayofweek
    weekend = ds.loc[ds["dow"] >= 5, "units_i"].mean() / ds.loc[ds["dow"] < 5, "units_i"].mean()
    signals.append(("7. Weekend demand uplift visible", weekend > 1.05, f"weekend ÷ weekday units = {weekend:.2f}"))
    feats = g["gld_demand_features"].copy()
    feats["rain"] = pd.to_numeric(feats["rainfall_mm"].replace("", np.nan))
    feats["t"] = feats["target_units"].astype(int)
    by_day = feats.groupby(["store_id", "business_date"]).agg(t=("t", "sum"), rain=("rain", "first"))
    rain_ratio = by_day.loc[by_day["rain"] > 10, "t"].mean() / by_day.loc[by_day["rain"] == 0, "t"].mean()
    signals.append(("8. Rain demand uplift visible", rain_ratio > 1.0, f"rainy (>10 mm) ÷ dry store-day units = {rain_ratio:.2f}"))

    gen_root = resolve(paths["generation"])
    loads = read_meta(resolve(paths["metadata"]), "meta_file_loads")
    gen_dir = gen_root / loads.loc[loads["status"] == "loaded", "generation_run_id"].iloc[0]
    planted = read_csv_strings(gen_dir / "ground_truth" / "gt_affinity_pairs.csv")
    bp = g["gld_basket_pairs"]
    bp["lift_f"] = bp["lift"].astype(float)
    top = set(map(frozenset, bp.nlargest(max(50, 2 * len(planted)), "lift_f")[["product_a", "product_b"]].values))
    hits = sum(frozenset(p) in top for p in planted[["antecedent_product_id", "consequent_product_id"]].values)
    signals.append(("9. Planted product pairs rank at the top by lift", hits >= 0.8 * len(planted),
                    f"{hits} of {len(planted)} planted pairs in the top {max(50, 2 * len(planted))}"))

    gt = read_csv_strings(gen_dir / "ground_truth" / "gt_stockouts.csv")
    truth_days = set()
    for r in gt.itertuples():
        start = (pd.Timestamp(r.start_ts) + pd.Timedelta(hours=5, minutes=30)).normalize()
        end = (pd.Timestamp(r.end_ts) + pd.Timedelta(hours=5, minutes=30)).normalize()
        for d in pd.date_range(start, end):
            truth_days.add((r.store_id, r.product_id, d.strftime("%Y-%m-%d")))
    flagged_days = set(map(tuple, inv.loc[inv["is_stockout"] == "true", ["store_id", "product_id", "business_date"]].values))
    recall = len(truth_days & flagged_days) / len(truth_days) if truth_days else 1.0
    precision = len(truth_days & flagged_days) / len(flagged_days) if flagged_days else 1.0
    info["stockout_recall"], info["stockout_precision"] = recall, precision
    signals.append(("10. Real stockout days detected from rebuilt inventory", recall >= 0.6,
                    f"recall {recall:.0%}, precision {precision:.0%} vs the simulator's true stockout days"))
    return core, signals, info


def main() -> int:
    paths = load_config("small").paths
    run = read_meta(resolve(paths["metadata"]), "meta_pipeline_runs")
    profile = run.sort_values("started_at")["profile"].iloc[-1]
    core, signals, _ = evaluate(paths, profile)
    print("\nGold completion report\n\nCore checks:")
    for name, ok, detail in core:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {detail}")
    print("\nPlanted-signal checks:")
    for name, ok, detail in signals:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {detail}")
    passed = sum(ok for _, ok, _ in core + signals)
    total = len(core) + len(signals)
    print(f"\n{passed}/{total} checks passed -> Gold is {'COMPLETE' if passed == total else 'NOT complete'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
