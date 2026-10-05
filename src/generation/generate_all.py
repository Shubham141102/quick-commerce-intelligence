"""Generate all source datasets for one run.

    python -m src.generation.generate_all --profile small
    python -m src.generation.generate_all --profile medium --seed 7

Output (all CSV):
    data/generation/<run_id>/landing/{historical,batch,stream}/<dataset>/*.csv
    data/generation/<run_id>/manifest/manifest_{run,datasets,issues,files}.csv
    data/generation/<run_id>/ground_truth/*.csv     (evaluation only, never read by the pipeline)
    data/generation/LATEST                          (id of the newest run)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.io import to_source_strings
from src.common.paths import resolve
from src.common.schemas import DIMENSION_DATASETS, SOURCE_SCHEMAS
from src.generation.anomalies import plan_anomalies
from src.generation.context import IST_OFFSET, GenContext
from src.generation.customers import build_customers
from src.generation.dirty_data import InjectionResult, Lookups, inject, plan_issue_counts
from src.generation.downstream import build_transactions
from src.generation.events import build_application_events, build_application_logs
from src.generation.inventory import simulate_inventory
from src.generation.landing import RAW, SLICE_TS, write_landing
from src.generation.manifest import write_ground_truth, write_manifest
from src.generation.orders import build_order_skeleton, fill_baskets
from src.generation.reference import build_reference
from src.generation.weather import build_weather


@dataclass
class GenerationResult:
    run_id: str
    cfg: Config
    out_dir: Path | None
    clean: dict[str, pd.DataFrame]                     # typed clean frames (source columns + helpers)
    injected: dict[str, InjectionResult] = field(default_factory=dict)
    datasets: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    ground_truth: dict[str, list[dict]] = field(default_factory=dict)


def _ist(ts: np.ndarray) -> np.ndarray:
    return ts.astype("datetime64[s]") + IST_OFFSET


def build_clean_tables(ctx: GenContext) -> dict[str, pd.DataFrame]:
    """Run every generator; each frame has the source columns plus `_slice_ts` (IST)."""
    ref = build_reference(ctx)
    cust = build_customers(ctx)
    weather = build_weather(ctx)
    plan = plan_anomalies(ctx, ref)
    skeleton = build_order_skeleton(ctx, ref, cust, weather, plan)
    book = fill_baskets(ctx, ref, cust, skeleton, plan)
    inv = simulate_inventory(ctx, ref, book)
    tx = build_transactions(ctx, ref, cust, weather, plan, book, inv)
    app_events = build_application_events(ctx, ref, tx)
    logs = build_application_logs(ctx, ref, tx, plan)

    period_start = ctx.days[0].astype("datetime64[s]")
    order_ist = _ist(tx.orders["order_ts"].to_numpy())
    tables = {
        "categories": ref.categories,
        "products": ref.products,
        "stores": ref.stores,
        "customers": cust.customers,
        "delivery_partners": ref.partners,
        "promotions": ref.promotions,
        "orders": tx.orders,
        "order_items": tx.order_items,
        "payments": tx.payments,
        "deliveries": tx.deliveries,
        "cancellations": tx.cancellations,
        "inventory_snapshots": inv.snapshots,
        "inventory_events": inv.events,
        "application_logs": logs,
        "order_promotions": tx.order_promotions,
        "returns_refunds": tx.returns_refunds,
        "reviews": tx.reviews,
        "application_events": app_events,
        "weather": weather.weather,
    }
    slice_source = {
        "orders": order_ist,
        "order_items": order_ist[tx.order_items["order_idx"].to_numpy()],
        "order_promotions": order_ist[tx.order_promotions["order_idx"].to_numpy()],
        "payments": _ist(tx.payments["payment_ts"].to_numpy()),
        "deliveries": _ist(tx.deliveries["pickup_ts"].to_numpy()),
        "cancellations": _ist(tx.cancellations["cancelled_ts"].to_numpy()),
        "inventory_snapshots": _ist(inv.snapshots["snapshot_ts"].to_numpy()),
        "inventory_events": _ist(inv.events["event_ts"].to_numpy()),
        "application_logs": _ist(logs["event_ts"].to_numpy()),
        "returns_refunds": _ist(tx.returns_refunds["refund_ts"].to_numpy()),
        "reviews": _ist(tx.reviews["review_ts"].to_numpy()),
        "application_events": _ist(app_events["event_ts"].to_numpy()),
        "weather": _ist(weather.weather["observation_ts"].to_numpy()),
        "customers": np.maximum(cust.customers["signup_date"].to_numpy().astype("datetime64[s]"), period_start),
    }
    for name, frame in tables.items():
        frame = frame.copy()
        if name in DIMENSION_DATASETS:
            frame[SLICE_TS] = period_start
        else:
            frame[SLICE_TS] = slice_source[name]
        tables[name] = frame
    return tables


def build_lookups(clean: dict[str, pd.DataFrame]) -> Lookups:
    orders, payments, partners = clean["orders"], clean["payments"], clean["delivery_partners"]
    lk = Lookups()
    for col, (table, key) in {
        "customer_id": ("customers", "customer_id"), "store_id": ("stores", "store_id"),
        "product_id": ("products", "product_id"), "order_id": ("orders", "order_id"),
        "partner_id": ("delivery_partners", "partner_id"), "payment_id": ("payments", "payment_id"),
        "promotion_id": ("promotions", "promotion_id"),
    }.items():
        lk.valid_ids[col] = set(clean[table][key].tolist())
    order_ts = pd.to_datetime(orders["order_ts"])
    lk.order_ts = dict(zip(orders["order_id"], order_ts.dt.to_pydatetime()))
    lk.order_date_ist = dict(zip(orders["order_id"], (order_ts + pd.Timedelta(hours=5, minutes=30)).dt.strftime("%Y-%m-%d")))
    lk.payment_amount = dict(zip(payments["payment_id"], payments["amount"].astype(float)))
    lk.payment_ts = dict(zip(payments["payment_id"], pd.to_datetime(payments["payment_ts"]).dt.to_pydatetime()))
    promos = clean["promotions"]
    lk.promotions = list(zip(promos["promotion_id"], pd.to_datetime(promos["start_date"]).dt.strftime("%Y-%m-%d"),
                             pd.to_datetime(promos["end_date"]).dt.strftime("%Y-%m-%d")))
    lk.partner_city = dict(zip(partners["partner_id"], partners["city_id"]))
    for pid, city in sorted(lk.partner_city.items()):
        lk.partners_by_city.setdefault(city, []).append(pid)
    items = clean["order_items"].merge(orders[["order_id", "customer_id"]], on="order_id")
    for cid, group in items.groupby("customer_id", sort=True)["product_id"]:
        lk.customer_products[cid] = set(group)
    lk.product_ids = clean["products"]["product_id"].tolist()
    return lk


def generate(profile: str = "small", seed: int | None = None, output_root: str | Path | None = None,
             write: bool = True, verbose: bool = True) -> GenerationResult:
    t0 = time.perf_counter()
    cfg = load_config(profile, seed)
    ctx = GenContext(cfg)
    run_id = f"gen_{cfg.profile}_s{cfg.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    root = resolve(output_root or cfg.paths["generation"])
    out_dir = root / run_id if write else None

    def log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)

    log(f"run {run_id}: generating clean data (profile={cfg.profile}, seed={cfg.seed})")
    clean = build_clean_tables(ctx)
    lookups = build_lookups(clean)
    result = GenerationResult(run_id, cfg, out_dir, clean, ground_truth=ctx.ground_truth)

    log("injecting dirty data and writing landing files")
    injected_rows: list[dict] = []
    for name, schema in SOURCE_SCHEMAS.items():
        strings = to_source_strings(clean[name], schema, keep=(SLICE_TS,))
        res = inject(name, strings, schema, cfg.dirty, lookups, ctx.rng("dirty", name))
        result.injected[name] = res
        injected_rows.extend(res.issue_rows)
        late = 0
        if write:
            frame = res.rows.copy()
            frame[RAW] = None
            for pos, line in res.raw_lines.items():
                frame.at[pos, RAW] = line
            stats = write_landing(ctx, name, frame, schema, out_dir / "landing")
            result.files.extend(stats.files)
            late = stats.late_rows
        planned = _planned(name, res, cfg)
        result.datasets.append({
            "dataset": name, "tier": schema.tier, "target_rows": ctx.targets[name],
            "generated_rows": res.original_rows, "duplicate_rows": res.duplicate_rows,
            "final_rows": res.final_rows, "clean_rows": res.clean_rows,
            "dirty_rows": res.final_rows - res.clean_rows,
            "clean_pct": round(100 * res.clean_rows / res.final_rows, 2) if res.final_rows else 100.0,
            "late_rows": late,
            "files": sum(1 for f in result.files if f["dataset"] == name),
        })
        for code in sorted(set(planned) | set(res.issue_counts)):
            result.issues.append({"dataset": name, "issue_code": code, "planned": planned.get(code, 0),
                                  "injected": res.issue_counts.get(code, 0)})

    if write:
        write_manifest(out_dir, run_id, cfg, result.datasets, result.issues, result.files)
        write_ground_truth(out_dir, ctx.ground_truth, injected_rows)
        (root / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    totals = pd.DataFrame(result.datasets)
    log(f"done: {totals['generated_rows'].sum():,} generated + {totals['duplicate_rows'].sum():,} duplicates "
        f"= {totals['final_rows'].sum():,} source rows, {100 * totals['clean_rows'].sum() / totals['final_rows'].sum():.2f}% clean"
        + (f", {len(result.files)} files -> {out_dir}" if write else ""))
    return result


def _planned(name: str, res: InjectionResult, cfg: Config) -> dict[str, int]:
    return plan_issue_counts(name, res.original_rows, cfg.dirty) if cfg.dirty.get("enabled", True) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic quick-commerce source data.")
    parser.add_argument("--profile", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--seed", type=int, default=None, help="override the seed in configs/default.yaml")
    parser.add_argument("--output-root", default=None, help="default: paths.generation from config")
    args = parser.parse_args()
    generate(args.profile, args.seed, args.output_root)


if __name__ == "__main__":
    main()
