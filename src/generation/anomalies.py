"""Planted anomalies: store outages, product demand spikes, payment-gateway failures.

They shape the generated data and are recorded in ground truth so anomaly
detection can be evaluated (Project_Plan_v2.md §8.6). The pipeline never reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.generation.context import GenContext, largest_remainder
from src.generation.reference import ReferenceData


@dataclass
class Outage:
    store_idx: int
    day_idx: int
    start_hour: int   # IST
    end_hour: int     # IST, exclusive


@dataclass
class Spike:
    store_idx: int
    product_idx: int
    day_from: int
    day_to: int       # inclusive


@dataclass
class GatewayFailure:
    start_utc: np.datetime64
    end_utc: np.datetime64


@dataclass
class AnomalyPlan:
    outages: list[Outage]
    spikes: list[Spike]
    gateway_failures: list[GatewayFailure]


def plan_anomalies(ctx: GenContext, ref: ReferenceData) -> AnomalyPlan:
    rng = ctx.rng("anomalies")
    n_stores, n_days = len(ref.stores), ctx.n_days
    counts = largest_remainder(ctx.cfg.planted.anomalies, {"outage": 0.4, "spike": 0.4, "gateway": 0.2})
    margin = min(3, max(0, n_days // 10))
    day_lo, day_hi = margin, n_days - margin  # keep anomalies away from the period edges

    # Spiked products: mid-popularity, never a focus SKU
    candidates = np.flatnonzero(
        (ref.products["popularity_rank"].to_numpy() > ctx.cfg.focus_skus)
        & (ref.products["popularity_rank"].to_numpy() <= max(ctx.cfg.focus_skus + 125, 50))
    )

    outages = [
        Outage(int(rng.integers(n_stores)), int(rng.integers(day_lo, day_hi)), h, h + int(rng.integers(2, 7)))
        for h in rng.integers(9, 17, size=counts["outage"]).tolist()
    ]
    spikes = []
    for _ in range(counts["spike"]):
        d = int(rng.integers(day_lo, max(day_lo + 1, day_hi - 1)))
        spikes.append(Spike(int(rng.integers(n_stores)), int(rng.choice(candidates)), d,
                            min(n_days - 1, d + int(rng.integers(0, 2)))))
    failures = []
    for _ in range(counts["gateway"]):
        d = int(rng.integers(day_lo, day_hi))
        start = ctx.day_start_utc(d) + np.timedelta64(int(rng.integers(10, 21)) * 3600, "s")
        failures.append(GatewayFailure(start, start + np.timedelta64(int(rng.integers(1, 4)) * 3600, "s")))

    store_ids = ref.stores["store_id"].to_numpy()
    product_ids = ref.products["product_id"].to_numpy()
    gt = []
    for o in outages:
        start = ctx.day_start_utc(o.day_idx) + np.timedelta64(o.start_hour * 3600, "s")
        gt.append({"anomaly_type": "store_outage", "store_id": store_ids[o.store_idx], "product_id": "",
                   "start_ts": start, "end_ts": start + np.timedelta64((o.end_hour - o.start_hour) * 3600, "s"),
                   "description": f"No orders for {o.end_hour - o.start_hour}h (store systems down)"})
    for s in spikes:
        gt.append({"anomaly_type": "demand_spike", "store_id": store_ids[s.store_idx],
                   "product_id": product_ids[s.product_idx],
                   "start_ts": ctx.day_start_utc(s.day_from),
                   "end_ts": ctx.day_start_utc(s.day_to) + np.timedelta64(86399, "s"),
                   "description": "Product appears in ~25% of the store's orders (viral demand)"})
    for f in failures:
        gt.append({"anomaly_type": "payment_gateway_failure", "store_id": "", "product_id": "",
                   "start_ts": f.start_utc, "end_ts": f.end_utc,
                   "description": "Elevated failed payment attempts and payment-service ERROR logs"})
    for i, row in enumerate(gt, start=1):
        row["anomaly_id"] = f"AN{i:03d}"
    ctx.ground_truth["gt_anomalies"] = gt
    return AnomalyPlan(outages, spikes, failures)
