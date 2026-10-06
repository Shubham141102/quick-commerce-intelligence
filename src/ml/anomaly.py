"""Sales anomaly detection (Phase 5C): three statistical detectors, one per kind of disruption.

Each detector compares an observed count with what is normal for the same store / hour / product over the
previous 28 days, and scores how unlikely the count is (p-value). A flag is a signal to investigate, not proof
of a problem.

- store_outage    : EXPERIMENTAL. Runs of 2–8 operating hours (06:00–23:59 IST) with far fewer orders than
                    expected at that store; p = P(X ≤ observed). Low volume (~1 order/hour) makes short outages
                    statistically invisible from orders alone — documented as a limitation.
- demand_spike    : orders containing a product at a store on a day; expected = the product's share of the
                    store's orders over the previous 28 days × that day's store orders (a busy day is not a
                    spike); p = P(X ≥ observed). Consecutive flagged days are merged.
- payment_failure : network-wide hourly failed payment attempts and payment-service ERROR logs far above
                    normal (Poisson); consecutive flagged hours are merged.

Revision (2026-10-05, after the first measurement): order counts vary more than Poisson allows (store-day
variance 27.7 vs mean 15.6 — weekdays, rain, promotions), which produced many false alarms. Outages and spikes
now use a negative binomial whose extra variation α is estimated from the data (var = μ + α·μ²); thresholds
were not changed. See docs/09_business_workspace.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from src.common.io import read_csv_strings
from src.ml.tables import ANOMALY_SERIES, SALES_ANOMALIES, write_table
from src.orchestration.tracking import RunTracker, utc_now

LOOKBACK_DAYS = 28
IST = pd.Timedelta(hours=5, minutes=30)
OPEN_HOURS = range(6, 24)
OUTAGE_P, OUTAGE_MIN_EXPECTED, OUTAGE_WINDOWS = 1e-3, 6.0, range(2, 9)
SPIKE_P, SPIKE_MIN_ORDERS = 1e-4, 4
PAYMENT_P, PAYMENT_MIN_COUNT = 1e-6, 4
MIN_EXPECTED = 0.05


def _read(silver_root: Path, dataset: str, columns: list[str]) -> pd.DataFrame:
    parts = sorted((silver_root / f"slv_{dataset}").glob("*.csv"))
    return pd.concat([read_csv_strings(p)[columns] for p in parts], ignore_index=True)


def _ist(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts.str.replace("Z", "", regex=False)) + IST


def _baseline(matrix: np.ndarray, day_type: np.ndarray) -> np.ndarray:
    """Mean of the previous LOOKBACK_DAYS days of the same day type (weekday / weekend), per cell.
    matrix: [..., day, hour] counts. Returns the same shape (NaN where there is no history)."""
    out = np.full(matrix.shape, np.nan)
    n_days = matrix.shape[-2]
    for d in range(n_days):
        past = [p for p in range(max(0, d - LOOKBACK_DAYS), d) if day_type[p] == day_type[d]]
        if len(past) >= 3:
            out[..., d, :] = matrix[..., past, :].mean(axis=-2)
    return out


def nb_alpha(obs: np.ndarray, exp: np.ndarray) -> float:
    """Method-of-moments extra variation α in var = μ + α·μ² (0 = Poisson)."""
    ok = np.isfinite(exp) & (exp > 0)
    o, e = obs[ok], exp[ok]
    return max(0.0, float(((o - e) ** 2 - e).sum() / (e ** 2).sum())) if len(e) else 0.0


def _nb(alpha: float, mu):
    mu = np.asarray(mu, dtype=float)
    return nbinom(1 / alpha, 1 / (1 + alpha * mu)) if alpha > 1e-6 else poisson(mu)


def detect_outages(orders: pd.DataFrame, days: pd.DatetimeIndex, stores: list[str]):
    ts = _ist(orders["order_ts"])
    df = pd.DataFrame({"store_id": orders["store_id"], "day": ts.dt.normalize(), "hour": ts.dt.hour})
    counts = np.zeros((len(stores), len(days), 24))
    s_idx, d_idx = {s: i for i, s in enumerate(stores)}, {d: i for i, d in enumerate(days)}
    for (s, d, h), n in df.groupby(["store_id", "day", "hour"]).size().items():
        if s in s_idx and d in d_idx:
            counts[s_idx[s], d_idx[d], h] = n
    day_type = (days.dayofweek >= 5).astype(int)
    expected = _baseline(counts, day_type)
    hours = list(OPEN_HOURS)
    # α from operating-hour store-day totals: a slow day lowers every hour of that day together
    alpha = nb_alpha(counts[:, :, hours].sum(axis=2), expected[:, :, hours].sum(axis=2))
    found, series = [], []
    for si, store in enumerate(stores):
        for di, day in enumerate(days):
            exp = expected[si, di, hours]
            if np.isnan(exp).any():
                continue
            obs = counts[si, di, hours]
            best = None
            for width in OUTAGE_WINDOWS:
                for start in range(len(hours) - width + 1):
                    o, e = obs[start:start + width].sum(), exp[start:start + width].sum()
                    if e < OUTAGE_MIN_EXPECTED:
                        continue
                    p = float(_nb(alpha, e).cdf(o))
                    if best is None or p < best[0]:
                        best = (p, start, width, o, e)
            if best and best[0] < OUTAGE_P:
                p, start, width, o, e = best
                h0 = hours[start]
                key = len(found)
                found.append({"detector": "store_outage", "store_id": store, "product_id": None, "business_date": day,
                              "start_ts": day + pd.Timedelta(hours=h0) - IST,
                              "end_ts": day + pd.Timedelta(hours=h0 + width) - IST,
                              "observed": o, "expected": e, "p_value": p,
                              "description": f"(experimental) {int(o)} orders between {h0:02d}:00 and {h0 + width:02d}:00 IST "
                                             f"where {e:.1f} were expected"})
                series += [{"_key": key, "ts": day + pd.Timedelta(hours=h) - IST, "observed": counts[si, di, h],
                            "expected": expected[si, di, h]} for h in hours]
    return found, series


def detect_spikes(orders: pd.DataFrame, items: pd.DataFrame, days: pd.DatetimeIndex):
    ts = _ist(orders["order_ts"])
    o = pd.DataFrame({"order_id": orders["order_id"], "store_id": orders["store_id"], "day": ts.dt.normalize()})
    lines = items[["order_id", "product_id"]].drop_duplicates().merge(o, on="order_id")
    daily = lines.groupby(["store_id", "product_id", "day"]).size().rename("orders").reset_index()
    full = (daily.set_index(["store_id", "product_id", "day"])["orders"]
            .unstack("day").reindex(columns=days, fill_value=0).fillna(0))
    store_day = (o.groupby(["store_id", "day"]).size().unstack("day").reindex(columns=days, fill_value=0).fillna(0))
    values = full.to_numpy()
    totals = store_day.reindex(full.index.get_level_values("store_id")).to_numpy()   # store orders per row
    cum = np.concatenate([np.zeros((len(values), 1)), np.cumsum(values, axis=1)], axis=1)
    cum_t = np.concatenate([np.zeros((len(totals), 1)), np.cumsum(totals, axis=1)], axis=1)
    lag = np.maximum(0, np.arange(len(days)) - LOOKBACK_DAYS)
    idx = np.arange(len(days))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (cum[:, idx] - cum[:, lag]) / (cum_t[:, idx] - cum_t[:, lag])   # share of store orders, past days
    expected = np.maximum(MIN_EXPECTED, np.nan_to_num(share) * totals)
    expected[:, :LOOKBACK_DAYS] = np.nan                                        # not enough history yet
    alpha = nb_alpha(values[:, LOOKBACK_DAYS:], expected[:, LOOKBACK_DAYS:])
    found, series = [], []
    for row, (store, product) in enumerate(full.index):
        flagged = []
        for d in range(LOOKBACK_DAYS, len(days)):
            obs, exp = values[row, d], expected[row, d]
            if obs < SPIKE_MIN_ORDERS:
                continue
            p = float(_nb(alpha, exp).sf(obs - 1))
            if p < SPIKE_P:
                flagged.append((d, obs, exp, p))
        for group in _runs(flagged):
            d0, d1 = group[0][0], group[-1][0]
            key = len(found)
            found.append({"detector": "demand_spike", "store_id": store, "product_id": product,
                          "business_date": days[d0], "start_ts": days[d0] - IST,
                          "end_ts": days[d1] + pd.Timedelta(days=1) - IST,
                          "observed": sum(g[1] for g in group), "expected": sum(g[2] for g in group),
                          "p_value": min(g[3] for g in group),
                          "description": f"{int(sum(g[1] for g in group))} orders contained this product over "
                                         f"{d1 - d0 + 1} day(s) where {sum(g[2] for g in group):.1f} were expected"})
            for d in range(max(0, d0 - 14), min(len(days), d1 + 15)):
                series.append({"_key": key, "ts": days[d] - IST, "observed": values[row, d],
                               "expected": expected[row, d] if np.isfinite(expected[row, d]) else np.nan})
    return found, series


def detect_payment_failures(payments: pd.DataFrame, logs: pd.DataFrame, days: pd.DatetimeIndex):
    def hourly(ts: pd.Series) -> np.ndarray:
        t = _ist(ts)
        m = np.zeros((len(days), 24))
        idx = {d: i for i, d in enumerate(days)}
        for (d, h), n in pd.DataFrame({"d": t.dt.normalize(), "h": t.dt.hour}).groupby(["d", "h"]).size().items():
            if d in idx:
                m[idx[d], h] = n
        return m

    failed = hourly(payments.loc[payments["status"] == "failed", "payment_ts"])
    errors = hourly(logs.loc[(logs["service"] == "payment-service") & (logs["log_level"] == "ERROR"), "event_ts"])
    day_type = np.zeros(len(days), dtype=int)  # payments behave the same on all days
    exp_f = _baseline(failed[None], day_type)[0]
    exp_e = _baseline(errors[None], day_type)[0]
    flagged = []
    for d in range(len(days)):
        for h in range(24):
            if np.isnan(exp_f[d, h]):
                continue
            pf = poisson.sf(failed[d, h] - 1, max(MIN_EXPECTED, exp_f[d, h])) if failed[d, h] >= PAYMENT_MIN_COUNT else 1
            pe = poisson.sf(errors[d, h] - 1, max(MIN_EXPECTED, exp_e[d, h])) if errors[d, h] >= PAYMENT_MIN_COUNT else 1
            p = min(pf, pe)
            if p < PAYMENT_P:
                flagged.append((d * 24 + h, failed[d, h], exp_f[d, h], p, errors[d, h]))
    found, series = [], []
    for group in _runs(flagged):
        k0, k1 = group[0][0], group[-1][0]
        d0 = k0 // 24
        key = len(found)
        start = days[d0] + pd.Timedelta(hours=k0 % 24) - IST
        found.append({"detector": "payment_failure", "store_id": None, "product_id": None, "business_date": days[d0],
                      "start_ts": start, "end_ts": start + pd.Timedelta(hours=k1 - k0 + 1),
                      "observed": sum(g[1] for g in group), "expected": sum(g[2] for g in group),
                      "p_value": min(g[3] for g in group),
                      "description": f"{int(sum(g[1] for g in group))} failed payments and "
                                     f"{int(sum(g[4] for g in group))} payment-service errors in {k1 - k0 + 1} hour(s), "
                                     f"{sum(g[2] for g in group):.1f} failures expected"})
        series += [{"_key": key, "ts": days[d0] + pd.Timedelta(hours=h) - IST, "observed": failed[d0, h],
                    "expected": exp_f[d0, h]} for h in range(24)]
    return found, series


def _runs(flagged: list[tuple]) -> list[list[tuple]]:
    """Group flags whose first element (day or hour index) is consecutive."""
    groups: list[list[tuple]] = []
    for f in flagged:
        if groups and f[0] == groups[-1][-1][0] + 1:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def run_anomalies(tracker: RunTracker, silver_root: Path, gold_root: Path, start, end) -> dict:
    t0, started = time.perf_counter(), utc_now()
    days = pd.date_range(start, end, freq="D")
    orders = _read(silver_root, "orders", ["order_id", "store_id", "order_ts"])
    items = _read(silver_root, "order_items", ["order_id", "product_id"])
    payments = _read(silver_root, "payments", ["status", "payment_ts"])
    logs = _read(silver_root, "application_logs", ["service", "log_level", "event_ts"])
    stores = sorted(orders["store_id"].unique())
    found, series = [], []
    for detect in (lambda: detect_outages(orders, days, stores), lambda: detect_spikes(orders, items, days),
                   lambda: detect_payment_failures(payments, logs, days)):
        f, s = detect()
        offset = len(found)
        found += f
        series += [{**row, "_key": row["_key"] + offset} for row in s]
    version = f"anomaly_{tracker.run_id}"
    anomalies = pd.DataFrame(found)
    if anomalies.empty:
        anomalies = pd.DataFrame(columns=[c.name for c in SALES_ANOMALIES.columns])
    anomalies = anomalies.sort_values(["start_ts", "detector"], kind="stable").reset_index(names="_key")
    anomalies["anomaly_id"] = [f"A{i + 1:04d}" for i in range(len(anomalies))]
    anomalies["score"] = (-np.log10(anomalies["p_value"].astype(float).clip(lower=1e-20))).round(2)
    anomalies["severity"] = np.where(anomalies["score"] >= 8, "High", "Medium")
    for col in ("observed", "expected"):
        anomalies[col] = anomalies[col].astype(float).round(2)
    anomalies["model_version"] = version
    ids = dict(zip(anomalies["_key"], anomalies["anomaly_id"]))
    ser = pd.DataFrame(series)
    ser["anomaly_id"] = ser["_key"].map(ids)
    ser["observed"], ser["expected"] = ser["observed"].astype(float), ser["expected"].astype(float).round(3)
    n = write_table(anomalies, gold_root, SALES_ANOMALIES)
    n_series = write_table(ser, gold_root, ANOMALY_SERIES)
    for table, rows in ((SALES_ANOMALIES, n), (ANOMALY_SERIES, n_series)):
        tracker.table(stage="ml", job=table.name, source=",".join(table.sources), target=table.name, engine="python",
                      module="src.ml.anomaly", rows_written=rows, status="success", started_at=started,
                      ended_at=utc_now(), duration_s=round(time.perf_counter() - t0, 2))
        for source in table.sources:
            tracker.lineage(source, table.name, table.name, engine="python")
    return {"anomalies": n, "by_detector": anomalies["detector"].value_counts().to_dict()}
