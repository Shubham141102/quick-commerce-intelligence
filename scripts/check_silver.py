"""Silver completion report (Project_Plan_v2.md §6.7) — pure pandas, no Spark.

    python -m scripts.check_silver

1. Row accounting: Bronze = Silver + quarantine + duplicates removed, per dataset.
2. No dirty values left: unique keys, no padded strings, allowed values only, required values present,
   canonical timestamp format.
3. Detected vs injected: every issue the generator injected (ground truth) is matched to what Silver
   did with that row — removed (DUP), fixed (FMT), quarantined (MISS/TYPE/NUM/FK/SEQ/MALF, refund BIZ)
   or flagged (other BIZ). The pipeline itself never reads ground truth; only this report does.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.common.schemas import SOURCE_SCHEMAS
from src.orchestration.tracking import read_meta
from src.transformations.silver.specs import CITY, SPECS

ISO_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
SOFT_FLAG = {"orders": "dq_total_mismatch", "order_items": "dq_unit_price_mismatch", "payments": "dq_amount_mismatch",
             "deliveries": "dq_partner_wrong_city", "order_promotions": "dq_promo_inactive",
             "reviews": "dq_unverified_purchase"}
QUARANTINE_CODES = {"MISS", "TYPE", "NUM", "FK", "SEQ", "MALF"}
# order_promotions has at most one row per order, so order_id identifies a row even when the
# injected issue replaced promotion_id (part of the business key).
MATCH_KEYS = {"order_promotions": ("order_id",)}


def _read_folder(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("*.csv"))
    return pd.concat([read_csv_strings(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def _key_strings(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    out = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    for i, k in enumerate(keys):
        out = out + ("" if i == 0 else ";") + f"{k}=" + frame[k].astype(str)
    return out


def evaluate(paths: dict, cities: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, bool, str]]]:
    silver_root, quarantine_root = resolve(paths["silver"]), resolve(paths["quarantine"])
    metadata = resolve(paths["metadata"])
    loads = read_meta(metadata, "meta_file_loads")
    gen_dir = resolve(paths["generation"]) / loads.loc[loads["status"] == "loaded", "generation_run_id"].iloc[0]
    truth = read_csv_strings(gen_dir / "ground_truth" / "gt_injected_issues.csv")
    runs = read_meta(metadata, "meta_table_runs")
    silver_run = runs[runs["stage"] == "silver"].sort_values("started_at")["run_id"].iloc[-1]
    table_runs = runs[(runs["run_id"] == silver_run)].set_index("job")

    checks: list[tuple[str, bool, str]] = []
    accounting, detection = [], []
    dirty_problems: list[str] = []
    dup_keys = false_flags = explained_flags = 0

    # Flags on rows the generator didn't touch can still be correct: a related row was damaged.
    def order_of(key: str) -> str:
        return key.split(";")[0].split("=", 1)[1]
    item_issues = truth[truth["dataset"] == "order_items"]
    detached = {order_of(k) for k in item_issues.loc[item_issues["mutation"].isin(["unknown_order_id", "order_id"]),
                                                      "business_key"]}   # an order lost one of its lines
    tampered = {order_of(k) for k in truth.loc[(truth["dataset"] == "orders") & (truth["issue_code"] == "BIZ"),
                                               "business_key"]}           # an order's own total was changed
    item_issue_orders = {order_of(k) for k in item_issues["business_key"]}
    explains = {"orders": detached, "payments": detached | tampered, "reviews": item_issue_orders}
    for ds, schema in SOURCE_SCHEMAS.items():
        spec, keys = SPECS[ds], schema.business_key
        silver = _read_folder(silver_root / f"slv_{ds}")
        qtn = _read_folder(quarantine_root / "qtn_records" / ds)
        tr = table_runs.loc[f"silver_{ds}"]
        bronze, dups = int(tr["rows_read"]), int(tr["rows_deduplicated"])
        cascade = int((qtn["rejection_type"] == "cascade").sum()) if len(qtn) else 0
        accounting.append({"dataset": ds, "bronze": bronze, "duplicates_removed": dups,
                           "quarantined_direct": len(qtn) - cascade, "quarantined_cascade": cascade,
                           "silver": len(silver), "balanced": "yes" if bronze == len(silver) + len(qtn) + dups else "NO"})

        # --- no dirty values left in Silver
        silver_keys = _key_strings(silver, keys)
        dup_keys += int(silver_keys.duplicated().sum())
        for col in schema.columns:
            values = silver[col.name]
            if col.dtype == "string" and (values != values.str.strip()).any():
                dirty_problems.append(f"{ds}.{col.name}: padded values")
            if not col.nullable and (values == "").any():
                dirty_problems.append(f"{ds}.{col.name}: empty required values")
            if col.dtype == "timestamp" and (~values[values != ""].str.match(ISO_RE)).any():
                dirty_problems.append(f"{ds}.{col.name}: non-ISO timestamps")
        for col, allowed in spec.enums.items():
            allowed_set = set(cities) if allowed == (CITY,) else set(allowed)
            if not set(silver[col][silver[col] != ""]) <= allowed_set:
                dirty_problems.append(f"{ds}.{col}: values outside the allowed set")

        # --- detected vs injected
        match = MATCH_KEYS.get(ds, keys)
        silver_match = _key_strings(silver, match)
        qtn_ids = qtn["source_record_id"] if len(qtn) else pd.Series(dtype=str)
        if match != keys and len(qtn):   # re-key quarantine rows on the match columns
            parts = qtn_ids.str.split(";", expand=True)
            qtn_ids = parts[[keys.index(k) for k in match]].agg(";".join, axis=1)
        silver_count = Counter(silver_match)
        direct_mask = (qtn["rejection_type"] == "direct") if len(qtn) else pd.Series(dtype=bool)
        qtn_direct = Counter(qtn_ids[direct_mask]) if len(qtn) else Counter()
        qtn_any = Counter(qtn_ids) if len(qtn) else Counter()
        flag_col = silver[SOFT_FLAG[ds]] if ds in SOFT_FLAG else pd.Series(dtype=str)
        flagged = set(silver_match[flag_col == "true"]) if ds in SOFT_FLAG else set()
        unverifiable = set(silver_match[flag_col == ""]) if ds in SOFT_FLAG else set()
        direct_rules = qtn.loc[direct_mask, "failed_rules"] if len(qtn) else pd.Series(dtype=str)
        rows = truth[truth["dataset"] == ds]
        if ds in SOFT_FLAG:
            injected_biz = {";".join(p for p in k.split(";") if p.split("=", 1)[0] in match)
                            for k in rows.loc[rows["issue_code"] == "BIZ", "business_key"]}
            uninjected = silver[(flag_col == "true") & ~silver_match.isin(injected_biz)]
            explained = uninjected["order_id"].isin(explains.get(ds, set())) if len(uninjected) else pd.Series(dtype=bool)
            explained_flags += int(explained.sum())
            false_flags += int((~explained).sum())
        for code, group in rows.groupby("issue_code"):
            handled = moot = not_verifiable = 0
            by_count: Counter = Counter()   # mutated key columns: matched by rule counts instead of key
            expect_quarantine = code in QUARANTINE_CODES or (code == "BIZ" and ds not in SOFT_FLAG)
            for _, issue in group.iterrows():
                key = issue["business_key"] if match == keys else ";".join(
                    p for p in issue["business_key"].split(";") if p.split("=", 1)[0] in match)
                mutated = issue["mutation"].removeprefix("unknown_")
                if expect_quarantine and mutated in match:
                    by_count[f"{'required_missing' if code == 'MISS' else 'fk_missing'}:{mutated}"] += 1
                    continue
                if not expect_quarantine and qtn_any[key] and not silver_count[key]:
                    moot += 1      # row rejected for another reason (parent rejected / parent unidentifiable)
                    continue
                if expect_quarantine and qtn_any[key] and not qtn_direct[key]:
                    moot += 1      # cascade-rejected: the parent was rejected first
                    continue
                if code == "DUP":
                    ok = silver_count[key] == 1
                elif code == "FMT":
                    ok = silver_count[key] == 1
                elif expect_quarantine:
                    ok = qtn_direct[key] >= 1 and silver_count[key] == 0
                elif key in unverifiable:
                    not_verifiable += 1
                    continue
                else:
                    ok = key in flagged
                handled += ok
            for rule, injected in by_count.items():
                handled += min(int(direct_rules.str.contains(rule, regex=False).sum()), injected)
            expected = len(group) - moot - not_verifiable
            outcome = ("removed" if code == "DUP" else "fixed" if code == "FMT" else
                       "flagged" if code == "BIZ" and ds in SOFT_FLAG else "quarantined")
            detection.append({"dataset": ds, "issue": code, "injected": len(group), "expected_outcome": outcome,
                              "handled": handled, "moot": moot, "not_verifiable": not_verifiable,
                              "rate": f"{100 * handled / expected:.1f}%" if expected else "n/a"})

    acc, det = pd.DataFrame(accounting), pd.DataFrame(detection)
    checks.append(("1. Every row accounted for (Bronze = Silver + quarantine + duplicates)",
                   (acc["balanced"] == "yes").all(), f"{acc['bronze'].sum():,} = {acc['silver'].sum():,} + "
                   f"{acc['quarantined_direct'].sum() + acc['quarantined_cascade'].sum():,} + {acc['duplicates_removed'].sum():,}"))
    checks.append(("2. Business keys unique in Silver", dup_keys == 0, f"{dup_keys} duplicate keys"))
    checks.append(("3. No dirty values left in Silver", not dirty_problems,
                   "clean" if not dirty_problems else "; ".join(dirty_problems[:5])))
    for outcome, label in [("removed", "4. Duplicates removed"), ("fixed", "5. Format issues fixed"),
                           ("quarantined", "6. Invalid rows quarantined"), ("flagged", "7. Business-rule issues flagged")]:
        part = det[det["expected_outcome"] == outcome]
        expected = (part["injected"] - part["moot"] - part["not_verifiable"]).sum()
        checks.append((label, part["handled"].sum() == expected, f"{part['handled'].sum():,} of {expected:,}"))
    checks.append(("7b. No unexplained business-rule flags", false_flags == 0,
                   f"{false_flags} unexplained; {explained_flags} caused by a damaged related row"))
    if len(table_runs):
        flags = sum((silver_root / f"slv_{ds}").exists() for ds in SOURCE_SCHEMAS)
        checks.append(("8. Silver table written for every dataset", flags == len(SOURCE_SCHEMAS), f"{flags}/{len(SOURCE_SCHEMAS)}"))
    edges = read_meta(metadata, "meta_lineage_edges")
    edges = edges[edges["run_id"] == silver_run]
    checks.append(("9. Lineage recorded (Bronze -> Silver, parent -> child)",
                   {f"slv_{d}" for d in SOURCE_SCHEMAS} <= set(edges["target"]), f"{len(edges)} edges"))
    return acc, det, checks


def main() -> int:
    cfg = load_config("small")
    acc, det, checks = evaluate(cfg.paths, tuple(c["city_id"] for c in load_config("medium").model_dump()["cities"]))
    pd.set_option("display.width", 200)
    print("\nSilver completion report\n\nRow accounting:")
    print(acc.to_string(index=False))
    print("\nDetected vs injected (ground truth):")
    print(det.to_string(index=False))
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:68s} {detail}")
    passed = sum(ok for _, ok, _ in checks)
    print(f"\n{passed}/{len(checks)} checks passed -> Silver is {'COMPLETE' if passed == len(checks) else 'NOT complete'}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
