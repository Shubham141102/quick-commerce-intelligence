"""Anomaly detectors on hand-made data: each catches its planted event and stays quiet on normal days."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from src.ml.anomaly import detect_outages, detect_payment_failures, detect_spikes  # noqa: E402

DAYS = pd.date_range("2025-08-01", periods=45, freq="D")
OUTAGE_DAY = 40


def _utc(day: pd.Timestamp, hour: int, minute: int = 10) -> str:
    return (day + pd.Timedelta(hours=hour, minutes=minute) - pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_outage_found_when_a_busy_store_goes_silent():
    rows = []
    for di, day in enumerate(DAYS):
        for hour in range(6, 24):
            if di == OUTAGE_DAY and 10 <= hour < 14:
                continue                                    # 4 silent hours at ~3 orders/hour
            rows += [{"order_id": f"O{di}-{hour}-{k}", "store_id": "S01", "order_ts": _utc(day, hour, 5 + k)} for k in range(3)]
    found, series = detect_outages(pd.DataFrame(rows), DAYS, ["S01"])
    assert len(found) == 1
    assert found[0]["business_date"] == DAYS[OUTAGE_DAY] and found[0]["observed"] == 0
    assert series and all(s["_key"] == 0 for s in series)


def _store_days(p9_per_day, other_per_day):
    orders, items = [], []
    for di, day in enumerate(DAYS):
        for k in range(p9_per_day(di)):
            orders.append({"order_id": f"P{di}-{k}", "store_id": "S02", "order_ts": _utc(day, 12)})
            items.append({"order_id": f"P{di}-{k}", "product_id": "P9"})
        for k in range(other_per_day(di)):                    # the rest of the store's orders
            orders.append({"order_id": f"X{di}-{k}", "store_id": "S02", "order_ts": _utc(day, 13)})
            items.append({"order_id": f"X{di}-{k}", "product_id": f"Q{k % 7}"})
    return pd.DataFrame(orders), pd.DataFrame(items)


def test_spike_found_and_consecutive_days_merged():
    rng = np.random.default_rng(0)
    p9 = [8 if di in (41, 42) else int(rng.random() < 0.3) for di in range(len(DAYS))]   # normally ~0.3 a day
    orders, items = _store_days(lambda di: p9[di], lambda di: 15)
    found, _ = detect_spikes(orders, items, DAYS)
    p9_found = [f for f in found if f["product_id"] == "P9"]
    assert len(p9_found) == 1 and p9_found[0]["observed"] == 16 and p9_found[0]["business_date"] == DAYS[41]


def test_busy_store_day_is_not_a_spike():
    rng = np.random.default_rng(2)
    p9 = [4 if di == 40 else int(rng.random() < 0.3) for di in range(len(DAYS))]
    orders, items = _store_days(lambda di: p9[di], lambda di: 60 if di == 40 else 15)
    found, _ = detect_spikes(orders, items, DAYS)     # day 40: the whole store is 4× busier, P9 rises with it
    assert not [f for f in found if f["product_id"] == "P9"]


def test_payment_failure_found_from_error_burst_and_failures():
    payments = [{"status": "success", "payment_ts": _utc(d, 12)} for d in DAYS]
    payments += [{"status": "failed", "payment_ts": _utc(DAYS[30], 18, m)} for m in range(0, 50, 7)]
    logs = [{"service": "payment-service", "log_level": "ERROR", "event_ts": _utc(DAYS[30], 18, m)} for m in range(40)]
    found, _ = detect_payment_failures(pd.DataFrame(payments), pd.DataFrame(logs), DAYS)
    assert len(found) == 1 and found[0]["business_date"] == DAYS[30]


def test_quiet_data_raises_nothing():
    rng = np.random.default_rng(1)
    rows = [{"order_id": f"O{di}-{h}-{k}", "store_id": "S01", "order_ts": _utc(day, h, k)}
            for di, day in enumerate(DAYS) for h in range(6, 24) for k in range(rng.poisson(3))]
    found, _ = detect_outages(pd.DataFrame(rows), DAYS, ["S01"])
    assert found == []
