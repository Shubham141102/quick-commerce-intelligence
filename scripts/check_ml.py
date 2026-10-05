"""Phase 4 ML completion report — pure pandas.

    python -m scripts.check_ml

4B demand forecasting: prediction grain and coverage, beats both baselines, honest uncertainty bands,
no train/test overlap, registry + loadable model files, planted demand drivers recognised.
4C stockout risk + replenishment: complete tiers with reasons, replenishment formula, backtest vs a naive
rule, planted supplier outages warned (ground truth read only here; pass bar fixed before measuring).
4D app: snapshot published within limits from the current model; Inventory data functions role-checked.
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.paths import resolve
from src.ml.tables import (
    DEMAND_PREDICTIONS,
    FEATURE_IMPORTANCE,
    FORECAST_METRICS,
    REPLENISHMENT,
    SKU_FORECAST,
    STOCKOUT_BACKTEST,
    STOCKOUT_RISK,
    read_table,
)
from src.orchestration.tracking import read_meta


def evaluate(paths: dict, strict_accuracy: bool = True, include_app: bool = True) -> list[tuple[str, bool, str]]:
    gold, metadata = resolve(paths["gold"]), resolve(paths["metadata"])
    preds, sku = read_table(gold, DEMAND_PREDICTIONS), read_table(gold, SKU_FORECAST)
    metrics, imp = read_table(gold, FORECAST_METRICS), read_table(gold, FEATURE_IMPORTANCE)
    checks: list[tuple[str, bool, str]] = []

    dup = int(preds.duplicated(list(DEMAND_PREDICTIONS.grain)).sum() + sku.duplicated(list(SKU_FORECAST.grain)).sum())
    bt, fut = preds[preds["run_type"] == "backtest"], preds[preds["run_type"] == "future"]
    series = preds.groupby(["store_id", "category_id"]).ngroups
    last_known = bt["forecast_date"].max()
    ok = (dup == 0 and bt["actual_units"].notna().all() and fut["actual_units"].isna().all()
          and (fut["forecast_date"] > last_known).all() and len(fut) == series * 7)
    checks.append(("1. Predictions complete: backtest has actuals, 7 future days per series", ok,
                   f"{len(bt):,} backtest + {len(fut):,} future rows for {series} series; {dup} duplicate keys"))

    pooled = metrics[(metrics["segment"] == "all") & (metrics["horizon"] == 0)].set_index(["evaluation", "model"])["wape"]
    cat = {m: pooled[("category_daily", m)] for m in ("model", "baseline_moving_avg", "baseline_seasonal_naive")}
    sku_w = {m: pooled[("sku_daily", m)] for m in ("model", "baseline_moving_avg")}
    beats = cat["model"] < min(cat["baseline_moving_avg"], cat["baseline_seasonal_naive"]) and \
        sku_w["model"] < sku_w["baseline_moving_avg"]
    checks.append(("2. Forecast beats both baselines (WAPE, all horizons)", beats or not strict_accuracy,
                   f"category: model {cat['model']:.3f} vs moving avg {cat['baseline_moving_avg']:.3f} vs seasonal naive "
                   f"{cat['baseline_seasonal_naive']:.3f}; SKU: {sku_w['model']:.3f} vs {sku_w['baseline_moving_avg']:.3f}"))

    cov = metrics[(metrics["model"] == "model") & (metrics["segment"] == "all") & (metrics["horizon"] == 0)
                  & (metrics["evaluation"] == "category_daily")]["interval_coverage"].iloc[0]
    ordered = bool(((preds["lower_units"] <= preds["forecast_units"] + 1e-9)
                    & (preds["forecast_units"] <= preds["upper_units"] + 1e-9)).all())
    checks.append(("3. Uncertainty band honest (lower ≤ forecast ≤ upper; ~80% coverage)",
                   ordered and (0.70 <= cov <= 0.95 or not strict_accuracy), f"coverage {cov:.1%} (target 80%)"))

    runs = read_meta(metadata, "meta_model_runs")
    runs = runs[runs["model_name"] == "demand_forecast"]
    latest = runs[runs["model_version"].str.startswith(preds["model_version"].iloc[0])]
    backtest = latest[latest["status"] == "evaluated"]
    no_overlap = len(backtest) == 1 and pd.Timestamp(backtest["train_end"].iloc[0]) < bt["forecast_date"].min()
    checks.append(("4. No train/test overlap (backtest model trained before September)", no_overlap,
                   f"trained up to {backtest['train_end'].iloc[0] if len(backtest) else '?'}, tested from "
                   f"{bt['forecast_date'].min().date()}"))

    loadable = []
    for path in latest["artifact_path"]:
        try:
            loadable.append(joblib.load(resolve(path))["model"].point is not None)
        except Exception:  # noqa: BLE001
            loadable.append(False)
    checks.append(("5. Model registry filled; model files load", len(latest) == 2 and all(loadable),
                   f"{len(latest)} registry rows (backtest + production), {sum(loadable)} files load"))

    rank = imp.set_index("feature")["rank"]
    drivers = rank.get("day_of_week", 99) <= 5 and rank.get("rainfall_mm", 99) <= 10 and rank.get("promo_active", 99) <= 10
    checks.append(("6. Planted demand drivers recognised (weekday, rain, promotion)", drivers or not strict_accuracy,
                   f"ranks: day_of_week {rank.get('day_of_week')}, promo_active {rank.get('promo_active')}, "
                   f"rainfall_mm {rank.get('rainfall_mm')} of {len(rank)}"))
    checks += _stockout_checks(paths, strict_accuracy)
    if include_app:
        checks += _app_checks(paths, preds["model_version"].iloc[0])
    return checks


def _app_checks(paths: dict, model_version: str) -> list[tuple[str, bool, str]]:
    import os

    from src.serving import queries as q
    from src.serving.db import get_snapshot
    from src.serving.permissions import AccessDenied

    demo = resolve(paths["demo"])
    out = []
    try:
        snap = get_snapshot(str(demo))
    except FileNotFoundError:
        return [("11. App snapshot published (< 50 MB, current model)", False, "no snapshot; run --stages publish")]
    info = snap.manifest
    ok = float(info["size_mb"]) < 50 and info["forecast_model_version"].startswith(model_version)
    out.append(("11. App snapshot published (< 50 MB, current model)", ok,
                f"{info['tables']} tables, {int(info['rows']):,} rows, {info['size_mb']} MB, run {info['pipeline_run_id']}"))
    previous = os.environ.get("QCI_DEMO_DIR")
    os.environ["QCI_DEMO_DIR"] = str(demo)
    get_snapshot.cache_clear()
    try:
        functions = [q.stores, q.categories, q.inventory_kpis, q.stock_health_by_store, q.forecast_accuracy,
                     q.accuracy_by_category, q.feature_importance, q.risk_list, q.replenishment_assumptions,
                     q.stockout_backtest]
        works = all(f("inventory_manager") is not None for f in functions)
        blocked = 0
        for f in functions:
            try:
                f("marketing_manager")
            except AccessDenied:
                blocked += 1
        out.append(("12. Inventory data functions work for the right role, block others", works and blocked == len(functions),
                    f"{len(functions)} functions ran for inventory_manager; {blocked}/{len(functions)} blocked for marketing_manager"))
    finally:
        if previous is None:
            os.environ.pop("QCI_DEMO_DIR", None)
        else:
            os.environ["QCI_DEMO_DIR"] = previous
        get_snapshot.cache_clear()
    return out


def _stockout_checks(paths: dict, strict: bool) -> list[tuple[str, bool, str]]:
    gold = resolve(paths["gold"])
    risk, repl, bt = read_table(gold, STOCKOUT_RISK), read_table(gold, REPLENISHMENT), read_table(gold, STOCKOUT_BACKTEST)
    out = []
    pairs = risk.groupby(["store_id", "product_id"]).ngroups
    per_day = risk.groupby("as_of_date").size()
    explained = risk.loc[risk["risk_tier"] != "Low", "reason_codes"].ne("").all()
    ok = (risk["risk_tier"].isin(["High", "Medium", "Low"]).all() and explained and (per_day == pairs).all()
          and (risk["run_type"] == "current").sum() == pairs)
    out.append(("7. Risk tiers for every focus SKU every day, each with a reason", ok,
                f"{len(per_day)} decision days × {pairs} SKUs; current: "
                + ", ".join(f"{k} {v}" for k, v in risk[risk['run_type'] == 'current']['risk_tier'].value_counts().items())))

    expected = np.ceil((repl["forecast_demand_protection"] + repl["safety_stock"] - repl["closing_stock"])
                       .clip(lower=0).round(6)).astype(int)
    consistent = bool((repl["suggested_qty"] == expected).all() and (repl["suggested_qty"] >= 0).all())
    current = repl[repl["run_type"] == "current"]
    out.append(("8. Replenishment follows its formula (never negative)", consistent,
                f"current: {int((current['suggested_qty'] > 0).sum())} SKUs to reorder, {int(current['suggested_qty'].sum())} units"))

    b = bt.set_index("rule")
    model_p, naive_p = b.loc["High tier (forecast-based)", "precision"], b.loc["Naive: stock <= reorder level", "precision"]
    base = b["stockouts_next_3d"].iloc[0] / b["pair_days"].iloc[0]
    out.append(("9. Forecast-based High tier more precise than the naive reorder rule", model_p > naive_p or not strict,
                f"precision {model_p:.1%} vs {naive_p:.1%} (base rate {base:.1%}); recall "
                f"{b.loc['High tier (forecast-based)', 'recall']:.1%} vs {b.loc['Naive: stock <= reorder level', 'recall']:.1%}"))

    # planted supplier outages (ground truth, read only here): warned High/Medium in the 3 days up to the start?
    loads = read_meta(resolve(paths["metadata"]), "meta_file_loads")
    gen_dir = resolve(paths["generation"]) / loads.loc[loads["status"] == "loaded", "generation_run_id"].iloc[0]
    episodes = pd.read_csv(gen_dir / "ground_truth" / "gt_stockout_episodes.csv")
    episodes["start_day"] = (pd.to_datetime(episodes["start_ts"].str.replace("Z", "")) + pd.Timedelta(hours=5, minutes=30)).dt.normalize()
    window = risk[risk["run_type"] == "backtest"]
    in_bt = episodes[episodes["start_day"].between(window["as_of_date"].min() + pd.Timedelta(days=3), window["as_of_date"].max())]
    warned = 0
    for e in in_bt.itertuples():
        w = window[(window["store_id"] == e.store_id) & (window["product_id"] == e.product_id)
                   & window["as_of_date"].between(e.start_day - pd.Timedelta(days=3), e.start_day)]
        warned += bool((w["risk_tier"] != "Low").any())
    rate = warned / len(in_bt) if len(in_bt) else 1.0
    out.append(("10. Planted supplier outages warned (High/Medium within 3 days before)", rate >= 0.5 or not strict,
                f"{warned} of {len(in_bt)} planted outages in the backtest window ({rate:.0%}; bar set at 50%)"))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = evaluate(load_config("small").paths)
    print("\nPhase 4 ML completion report\n")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:72s} {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
