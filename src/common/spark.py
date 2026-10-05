"""The one place a SparkSession is created (Project_Plan_v2.md §13.1).

On Windows, Spark needs Hadoop's native helpers (winutils.exe, hadoop.dll). They
live in the git-ignored `tools/hadoop/bin`; this module points HADOOP_HOME and PATH
at them before the JVM starts. It also pins Spark's Python workers to the current
interpreter, so they never pick up a different `python` from PATH.
"""

from __future__ import annotations

import os
import sys

import pyspark
from pyspark.sql import SparkSession

from src.common.paths import PROJECT_ROOT

HADOOP_HOME = PROJECT_ROOT / "tools" / "hadoop"


def _short_path(path: str) -> str:
    """Windows 8.3 short form of a path (no spaces); unchanged elsewhere or if unavailable."""
    if os.name != "nt":
        return path
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    return buf.value if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 1024) else path


def _configure_environment() -> None:
    # Spark's Windows .cmd launchers break on paths with spaces; a space-free SPARK_HOME avoids it
    os.environ.setdefault("SPARK_HOME", _short_path(os.path.dirname(pyspark.__file__)))
    python = _short_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = python
    os.environ["PYSPARK_DRIVER_PYTHON"] = python
    if os.name == "nt":
        if not os.environ.get("HADOOP_HOME"):
            if not (HADOOP_HOME / "bin" / "winutils.exe").exists():
                raise RuntimeError(f"winutils.exe not found in {HADOOP_HOME / 'bin'}; see README 'Spark on Windows'")
            os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
        bin_dir = str(os.path.join(os.environ["HADOOP_HOME"], "bin"))
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def get_spark(app_name: str = "quick-commerce-intelligence", master: str = "local[*]",
              shuffle_partitions: int = 8, driver_memory: str = "4g") -> SparkSession:
    _configure_environment()
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )
