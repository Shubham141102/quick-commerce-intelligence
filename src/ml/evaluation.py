"""Forecast accuracy metrics. WAPE is the headline metric (Project_Plan_v2.md §8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(actual: pd.Series, forecast: pd.Series, lower: pd.Series | None = None,
                     upper: pd.Series | None = None) -> dict[str, float]:
    a, f = actual.to_numpy(dtype=float), forecast.to_numpy(dtype=float)
    err = f - a
    total = a.sum()
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.abs(err).sum() / total) if total else float("nan"),
        "bias": float(err.sum() / total) if total else float("nan"),
        "rows": int(len(a)),
        "interval_coverage": float("nan"),
    }
    if lower is not None and upper is not None:
        lo, hi = lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)
        out["interval_coverage"] = float(np.mean((a >= lo) & (a <= hi)))
    return out
