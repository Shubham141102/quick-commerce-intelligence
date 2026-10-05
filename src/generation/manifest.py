"""Generation manifest and ground-truth files (Project_Plan_v2.md §3.8), all CSV."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import Config
from src.common.io import TIMESTAMP_FMT, write_csv
from src.common.paths import PROJECT_ROOT


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


def _write(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col]).dt.strftime(TIMESTAMP_FMT).fillna("")
    out = out.astype(object).where(out.notna(), "").astype(str)
    write_csv(out, path, list(out.columns))


def write_manifest(out_dir: Path, run_id: str, cfg: Config, datasets: list[dict], issues: list[dict],
                   files: list[dict]) -> None:
    run = pd.DataFrame([{
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime(TIMESTAMP_FMT),
        "generator_version": cfg.project.get("generator_version", ""),
        "git_commit": git_commit(),
        "profile": cfg.profile,
        "seed": cfg.seed,
        "start_date": cfg.calendar.start_date.isoformat(),
        "end_date": cfg.calendar.end_date.isoformat(),
        "clean_ratio_target": cfg.dirty["clean_ratio"],
        "config_hash": config_hash(cfg),
    }])
    _write(run, out_dir / "manifest" / "manifest_run.csv")
    _write(pd.DataFrame(datasets), out_dir / "manifest" / "manifest_datasets.csv")
    _write(pd.DataFrame(issues), out_dir / "manifest" / "manifest_issues.csv")
    _write(pd.DataFrame(files), out_dir / "manifest" / "manifest_files.csv")


def write_ground_truth(out_dir: Path, ground_truth: dict[str, list[dict]], injected: list[dict]) -> None:
    for name, rows in ground_truth.items():
        _write(pd.DataFrame(rows), out_dir / "ground_truth" / f"{name}.csv")
    _write(pd.DataFrame(injected, columns=["dataset", "issue_code", "mutation", "business_key"]),
           out_dir / "ground_truth" / "gt_injected_issues.csv")
