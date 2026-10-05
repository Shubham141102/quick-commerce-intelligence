"""Pipeline runner: resolves the generation run, creates the tracker, runs stages, records the outcome.

Stages are added phase by phase; Phase 2 starts with `ingest` (landing -> Bronze).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.config import Config, load_config
from src.common.io import read_csv_strings
from src.common.paths import resolve
from src.generation.manifest import git_commit
from src.orchestration.tracking import RunTracker

STAGES = ("ingest", "silver", "gold")


@dataclass
class PipelineResult:
    run_id: str
    status: str
    generation_dir: Path
    results: dict[str, dict] = field(default_factory=dict)


def resolve_generation_dir(generation_root: Path, generation_run: str | None) -> Path:
    run = generation_run or (generation_root / "LATEST").read_text(encoding="utf-8").strip()
    path = generation_root / run
    if not (path / "landing").is_dir():
        raise FileNotFoundError(f"no landing folder for generation run {run!r} under {generation_root}")
    return path


def generation_profile(generation_dir: Path) -> str:
    return read_csv_strings(generation_dir / "manifest" / "manifest_run.csv").at[0, "profile"]


def run_pipeline(stages: list[str], generation_run: str | None = None, generation_root: str | Path | None = None,
                 force_reload: bool = False, reset_bronze: bool = False, overrides: dict | None = None,
                 spark=None) -> PipelineResult:
    base = load_config("small", overrides=overrides)  # only for paths; profile comes from the generation run
    gen_root = resolve(generation_root or base.paths["generation"])
    generation_dir = resolve_generation_dir(gen_root, generation_run)
    cfg: Config = load_config(generation_profile(generation_dir), overrides=overrides)
    bronze_root, metadata_root = resolve(cfg.paths["bronze"]), resolve(cfg.paths["metadata"])

    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; available: {STAGES}")

    owns_spark = spark is None
    if owns_spark:
        from src.common.spark import get_spark
        spark = get_spark()
        spark.sparkContext.setLogLevel("ERROR")
    tracker = RunTracker(metadata_root, "quick_commerce_pipeline", cfg.profile, generation_dir.name,
                         spark_version=spark.version, git_commit=git_commit())
    result = PipelineResult(tracker.run_id, "running", generation_dir)
    try:
        if "ingest" in stages:
            from src.ingestion.bronze import reset_bronze as _reset
            from src.ingestion.bronze import run_ingest
            if reset_bronze:
                _reset(bronze_root, metadata_root)
            with tracker.stage("ingest"):
                result.results["ingest"] = run_ingest(spark, tracker, generation_dir, bronze_root,
                                                      metadata_root, force_reload)
        if "silver" in stages:
            from src.transformations.silver.run import run_silver
            if not any(bronze_root.glob("brz_*/batch=*")):
                raise FileNotFoundError("Bronze is empty; run the ingest stage first")
            with tracker.stage("silver"):
                result.results["silver"] = run_silver(
                    spark, tracker, bronze_root, resolve(cfg.paths["silver"]), resolve(cfg.paths["quarantine"]),
                    tuple(c.city_id for c in cfg.cities))
        if "gold" in stages:
            from src.transformations.gold.run import run_gold
            silver_root = resolve(cfg.paths["silver"])
            if not (silver_root / "slv_orders").exists():
                raise FileNotFoundError("Silver is empty; run the silver stage first")
            with tracker.stage("gold"):
                result.results["gold"] = run_gold(spark, tracker, cfg, silver_root, resolve(cfg.paths["gold"]),
                                                  metadata_root)
        result.status = "success"
        tracker.finish("success")
    except BaseException as exc:
        result.status = "failed"
        tracker.finish("failed", exc)
        raise
    finally:
        if owns_spark:
            spark.stop()
    return result
