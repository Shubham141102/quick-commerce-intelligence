import pytest

from src.generation.generate_all import generate


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    """One small-profile generation shared by all tests (written to a temp folder)."""
    return generate("small", output_root=tmp_path_factory.mktemp("generation"), verbose=False)


@pytest.fixture(scope="session")
def pipeline_env(small_run, spark, tmp_path_factory):
    """Small run ingested into a temporary Bronze; Silver tests build on it."""
    from pathlib import Path

    from src.orchestration.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("pipeline")
    gen_root = Path(small_run.out_dir).parent
    paths = {"generation": str(gen_root), "bronze": str(root / "bronze"), "silver": str(root / "silver"),
             "quarantine": str(root / "quarantine"), "gold": str(root / "gold"), "metadata": str(root / "metadata")}
    overrides = {"paths": paths}
    first = run_pipeline(["ingest"], small_run.run_id, gen_root, overrides=overrides, spark=spark)
    return {"root": root, "overrides": overrides, "paths": paths, "gen_root": gen_root, "first": first,
            "bronze": root / "bronze", "silver": root / "silver", "quarantine": root / "quarantine",
            "gold": root / "gold", "metadata": root / "metadata"}


@pytest.fixture(scope="session")
def silver_env(pipeline_env, small_run, spark):
    from src.orchestration.pipeline import run_pipeline

    result = run_pipeline(["silver"], small_run.run_id, pipeline_env["gen_root"],
                          overrides=pipeline_env["overrides"], spark=spark)
    return {**pipeline_env, "result": result}


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from src.common.spark import get_spark

    session = get_spark("qci-tests", master="local[2]", shuffle_partitions=4, driver_memory="2g")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
