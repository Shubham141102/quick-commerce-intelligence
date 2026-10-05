"""Export one flat CSV per table (for review / presentation).

    python -m scripts.export_flat_tables                       # Bronze -> data_sep/<dataset>.csv (raw, dirty rows included)
    python -m scripts.export_flat_tables --layer silver        # Silver -> data_sep_silver/<dataset>.csv + quarantine.csv
    python -m scripts.export_flat_tables --layer gold          # Gold   -> data_sep_gold/<table>.csv (12 tables)
    python -m scripts.export_flat_tables --with-metadata       # also keep the `_` lineage/metadata columns
    python -m scripts.export_flat_tables --out my_folder

Rows are sorted by the table's business key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_config
from src.common.io import read_csv_strings, write_csv
from src.common.paths import PROJECT_ROOT, resolve
from src.common.schemas import SOURCE_SCHEMAS, bronze_columns
from src.transformations.silver.engine import LINEAGE, output_columns
from src.transformations.silver.specs import SPECS

DEFAULT_OUT = {"bronze": "data_sep", "silver": "data_sep_silver", "gold": "data_sep_gold"}


def export_gold(gold_root: Path, out_dir: Path) -> pd.DataFrame:
    from src.transformations.gold.run import GOLD_TABLES

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for table in GOLD_TABLES:
        parts = sorted((gold_root / table.name).glob("*.csv"))
        if not parts:
            raise FileNotFoundError(f"no Gold files for {table.name}; run: python -m scripts.run_pipeline --stages gold")
        frame = _concat(parts)
        write_csv(frame, out_dir / f"{table.name}.csv", table.column_names)
        summary.append({"table": table.name, "rows": len(frame), "files_merged": len(parts),
                        "file": (out_dir / f"{table.name}.csv").relative_to(PROJECT_ROOT).as_posix()})
    return pd.DataFrame(summary)


def _concat(parts: list[Path]) -> pd.DataFrame:
    return pd.concat([read_csv_strings(p) for p in parts], ignore_index=True)


def export(layer_root: Path, out_dir: Path, layer: str = "bronze", with_metadata: bool = False,
           quarantine_root: Path | None = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset, schema in SOURCE_SCHEMAS.items():
        pattern = f"brz_{dataset}/batch=*/*.csv" if layer == "bronze" else f"slv_{dataset}/*.csv"
        parts = sorted(layer_root.glob(pattern))
        if not parts:
            stage = "ingest" if layer == "bronze" else "silver"
            raise FileNotFoundError(f"no {layer} files for {dataset}; run: python -m scripts.run_pipeline --stages {stage}")
        frame = _concat(parts).sort_values(list(schema.business_key), kind="stable")
        if layer == "bronze":
            columns = bronze_columns(dataset) if with_metadata else schema.names
        else:
            columns = [c for c in output_columns(SPECS[dataset]) if with_metadata or c not in LINEAGE]
        write_csv(frame, out_dir / f"{dataset}.csv", columns)
        summary.append({"table": dataset, "rows": len(frame), "files_merged": len(parts),
                        "file": (out_dir / f"{dataset}.csv").relative_to(PROJECT_ROOT).as_posix()})
    if layer == "silver" and quarantine_root is not None:
        qtn = _concat(sorted((quarantine_root / "qtn_records").glob("*/*.csv")))
        qtn = qtn.sort_values(["source_dataset", "rejection_type", "primary_rule", "source_record_id"], kind="stable")
        write_csv(qtn, out_dir / "quarantine.csv", list(qtn.columns))
        summary.append({"table": "(quarantine)", "rows": len(qtn), "files_merged": 19,
                        "file": (out_dir / "quarantine.csv").relative_to(PROJECT_ROOT).as_posix()})
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one flat CSV per table.")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold"], default="bronze")
    parser.add_argument("--out", default=None, help="output folder (default: data_sep / data_sep_silver)")
    parser.add_argument("--with-metadata", action="store_true", help="include `_` lineage/metadata columns")
    args = parser.parse_args()
    paths = load_config("small").paths
    out = resolve(args.out or DEFAULT_OUT[args.layer])
    if args.layer == "gold":
        summary = export_gold(resolve(paths["gold"]), out)
    else:
        summary = export(resolve(paths[args.layer]), out, args.layer, args.with_metadata, resolve(paths["quarantine"]))
    print(summary.to_string(index=False))
    print(f"\n{summary['rows'].sum():,} rows -> {out}")


if __name__ == "__main__":
    main()
