"""Bronze completion report: checks Bronze against its acceptance criteria (Project_Plan_v2.md §5).

    python -m scripts.check_bronze

Compares the Bronze tables with the generation run they were loaded from
(landing files + manifest) and the run metadata. Pure pandas; no Spark needed.
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.common.schemas import BRONZE_METADATA_COLUMNS, CORRUPT_RECORD, SOURCE_SCHEMAS
from src.orchestration.tracking import read_meta

REQUIRED_METADATA = ["_batch_id", "_pipeline_run_id", "_source_file", "_load_type", "_ingestion_ts", "_record_hash"]


def main() -> int:
    paths = load_config("small").paths
    bronze, metadata = resolve(paths["bronze"]), resolve(paths["metadata"])
    loads = read_meta(metadata, "meta_file_loads")
    loaded = loads[loads["status"] == "loaded"]
    if loaded.empty:
        print("Bronze is empty. Run: python -m scripts.run_pipeline --stages ingest")
        return 1
    gen_run = loaded["generation_run_id"].iloc[0]
    gen_dir = resolve(paths["generation"]) / gen_run
    manifest = read_csv_strings(gen_dir / "manifest" / "manifest_datasets.csv").set_index("dataset")
    files = read_csv_strings(gen_dir / "manifest" / "manifest_files.csv")
    issues = read_csv_strings(gen_dir / "ground_truth" / "gt_injected_issues.csv")

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))

    tables, per_table = {}, []
    for ds in SOURCE_SCHEMAS:
        parts = sorted((bronze / f"brz_{ds}").glob("batch=*/*.csv"))
        tables[ds] = pd.concat([read_csv_strings(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
        per_table.append({"table": f"brz_{ds}", "bronze_rows": len(tables[ds]),
                          "expected_rows": int(manifest.loc[ds, "final_rows"]), "batches": len(parts),
                          "corrupt_rows": int((tables[ds].get(CORRUPT_RECORD, pd.Series(dtype=str)) != "").sum())})
    report = pd.DataFrame(per_table)
    report["match"] = (report["bronze_rows"] == report["expected_rows"]).map({True: "yes", False: "NO"})

    check("1. Every source dataset has a Bronze table", all(len(t) for t in tables.values()),
          f"{sum(1 for t in tables.values() if len(t))}/{len(SOURCE_SCHEMAS)} tables")
    check("2. Row counts match the generation manifest", (report["match"] == "yes").all(),
          f"{report['bronze_rows'].sum():,} Bronze rows vs {report['expected_rows'].sum():,} expected")
    check("3. Every landing file loaded exactly once",
          loaded["file"].nunique() == len(files) and not loaded["file"].duplicated().any(),
          f"{loaded['file'].nunique():,} of {len(files):,} files, {int(loaded['file'].duplicated().sum())} duplicates")

    raw_ok, bad = True, []
    for ds, schema in SOURCE_SCHEMAS.items():
        expected = Counter()
        for f in files[files["dataset"] == ds]["file"]:
            expected.update(map(tuple, read_csv_strings(gen_dir / f)[schema.names].itertuples(index=False, name=None))
                            if ds != "application_logs" else [])
        if ds == "application_logs":
            continue  # malformed rows can't be compared column-by-column; checked in 5
        actual = Counter(map(tuple, tables[ds][schema.names].itertuples(index=False, name=None)))
        if actual != expected:
            raw_ok = False
            bad.append(ds)
    check("4. Source values preserved exactly (dirty values not fixed)", raw_ok,
          "all tables identical to landing files" if raw_ok else f"differences in {bad}")

    malf = int((issues["dataset"].eq("application_logs") & issues["issue_code"].eq("MALF")).sum())
    corrupt = int(report.loc[report["table"] == "brz_application_logs", "corrupt_rows"].iloc[0])
    check("5. Malformed rows kept, not dropped", corrupt == malf, f"{corrupt} corrupt log rows kept, {malf} injected")

    missing = {c: sum(int((t[c] == "").sum()) for t in tables.values()) for c in REQUIRED_METADATA}
    columns_ok = all(list(t.columns[-len(BRONZE_METADATA_COLUMNS):]) == list(BRONZE_METADATA_COLUMNS)
                     for t in tables.values())
    check("6. Ingestion metadata present on every row", columns_ok and not any(missing.values()),
          "all metadata columns filled" if not any(missing.values()) else f"empty values: {missing}")

    load_types = set(tables["orders"]["_load_type"])
    check("7. All ingestion patterns used (historical, batch, stream)", load_types == {"historical", "batch", "stream"},
          f"orders load types: {sorted(load_types)}")

    runs = read_meta(metadata, "meta_pipeline_runs")
    last = runs.sort_values("started_at").iloc[-1]
    check("8. Latest pipeline run succeeded", last["status"] == "success",
          f"{last['run_id']} -> {last['status']} ({last['duration_s']} s)")

    edges = read_meta(metadata, "meta_lineage_edges")
    check("9. Lineage recorded for every table", set(edges["target"]) >= {f"brz_{d}" for d in SOURCE_SCHEMAS},
          f"{edges[['source', 'target']].drop_duplicates().shape[0]} landing -> Bronze edges")

    print(f"\nBronze completion report  (generation run: {gen_run})\n")
    print(report.to_string(index=False))
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:62s} {detail}")
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} checks passed -> Bronze is {'COMPLETE' if passed == len(results) else 'NOT complete'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
