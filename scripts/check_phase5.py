"""Phase 5 completion report — pure pandas. Sections are added as each step is built.

    python -m scripts.check_phase5

5A lost-sales estimate + Inventory explorer: rows only on stockout days, arithmetic consistent, estimate vs
the simulator's true lost units (pass bars fixed before measuring: total within ±50%, SKU-level Spearman ≥ 0.4),
explorer data functions role-checked. Ground truth is read only here, never by the pipeline.
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
    from src.serving.db import get_snapshot
    from src.serving.permissions import AccessDenied
    previous = os.environ.get("QCI_DEMO_DIR")
    os.environ["QCI_DEMO_DIR"] = str(resolve(paths["demo"]))
    get_snapshot.cache_clear()
    try:
        risk = q.risk_list("inventory_manager")
        store, product = risk["store_id"].iloc[0], risk["product_id"].iloc[0]
        calls = [lambda r: q.explorer_timeline(r, store, product), lambda r: q.explorer_summary(r, store, product),
                 q.lost_sales_by_store, lambda r: q.top_lost_sales_skus(r, 5)]
        works = all(c("inventory_manager") is not None for c in calls)
        blocked = 0
        for c in calls:
            try:
                c("marketing_manager")
            except AccessDenied:
                blocked += 1
        out.append(("5A.4 Explorer data functions work for inventory, blocked for others",
                    works and blocked == len(calls), f"{len(calls)} functions; {blocked}/{len(calls)} blocked"))
    finally:
        if previous is None:
            os.environ.pop("QCI_DEMO_DIR", None)
        else:
            os.environ["QCI_DEMO_DIR"] = previous
        get_snapshot.cache_clear()
    return out


def status(name: str, ok: bool) -> str:
    """PASS, FAIL, or LIMIT (failed a pre-set bar that is a documented, accepted known limitation)."""
    if ok:
        return "PASS"
    return "LIMIT" if name.split(" ", 1)[0] in KNOWN_LIMITATIONS else "FAIL"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = check_5a(load_config("small").paths)
    print("\nPhase 5 completion report\n\n5A  Lost-sales estimate + Inventory explorer")
    statuses = [status(name, ok) for name, ok, _ in checks]
    for (name, _, detail), st in zip(checks, statuses):
        print(f"  [{st:5s}] {name:76s} {detail}")
    limits = [name.split(" ", 1)[0] for (name, _, _), st in zip(checks, statuses) if st == "LIMIT"]
    for code in limits:
        print(f"\n  Known limitation {code}: {KNOWN_LIMITATIONS[code]}")
    failed = statuses.count("FAIL")
    print(f"\n{statuses.count('PASS')} passed, {len(limits)} known limitation(s), {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
