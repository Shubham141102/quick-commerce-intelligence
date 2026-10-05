"""Application (clickstream) events and service logs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.generation.anomalies import AnomalyPlan
from src.generation.context import GenContext, make_ids
from src.generation.downstream import Transactions
from src.generation.orders import HOUR_W_DAY
from src.generation.reference import ReferenceData

SESSION_FLOW = ["app_open", "search", "view_product", "add_to_cart", "view_cart", "checkout_start", "payment_page"]
PLATFORMS = {"android": 0.62, "ios": 0.28, "web": 0.10}
APP_VERSIONS = ["5.2.0", "5.2.1", "5.3.0", "5.4.0"]
SEARCH_TERMS = ["milk", "bread", "chips", "eggs", "coke", "atta", "banana", "ice cream", "detergent", "paneer"]

SERVICES = ["order-service", "payment-service", "inventory-service", "delivery-service", "search-service"]
LEVEL_P = {"INFO": 0.80, "WARN": 0.12, "ERROR": 0.06, "DEBUG": 0.02}
LOG_TEMPLATES = {
    "INFO": ["Order {order} placed, items={n}, amount={amt}", "Payment captured for order {order}",
             "Rider assigned to order {order}, eta={eta} min", "Search served, query=\"{term}\", results={n}",
             "Stock updated for {product} at {store}"],
    "WARN": ["Slow response from gateway, latency={lat}ms, order={order}", "Low stock for {product} at {store}",
             "Rider delayed, order={order}, delay={eta} min", "Retrying search index refresh, attempt={n}"],
    "ERROR": ["Payment declined for order {order}, gateway=\"pg-sim\", code=502",
              "Failed to reserve stock for {product}, store={store}", "Delivery partner app timeout, order={order}",
              "Search backend unavailable, query=\"{term}\""],
    "DEBUG": ["Cache hit ratio={ratio}", "Feature flag checkout_v2={flag}"],
}


def build_application_events(ctx: GenContext, ref: ReferenceData, tx: Transactions) -> pd.DataFrame:
    """Sessions leading to orders (logged-in) plus anonymous browsing (customer_id empty)."""
    rng = ctx.rng("events", "app")
    target = ctx.targets["application_events"]
    anon_share = ctx.cfg.dirty["outside_budget"]["missing_optional"]["application_events.customer_id"]
    orders = tx.orders
    order_ts = orders["order_ts"].to_numpy().astype("datetime64[s]")
    product_ids = ref.products["product_id"].to_numpy()
    pop = ref.products["popularity_weight"].to_numpy()
    pop_cum = np.cumsum(pop)

    rows: list[tuple] = []
    session = 0
    while len(rows) < target:
        session += 1
        anonymous = rng.random() < anon_share
        if anonymous:
            day = int(rng.integers(ctx.n_days))
            hour = int(rng.choice(24, p=HOUR_W_DAY / HOUR_W_DAY.sum()))
            end_ts = ctx.day_start_utc(day) + np.timedelta64(hour * 3600 + int(rng.integers(3600)), "s")
            customer, steps = None, int(rng.integers(2, 5))
        else:
            i = int(rng.integers(len(orders)))
            end_ts, customer, steps = order_ts[i], orders.at[i, "customer_id"], int(rng.integers(3, 8))
        platform = str(rng.choice(list(PLATFORMS), p=list(PLATFORMS.values())))
        version = str(rng.choice(APP_VERSIONS))
        offsets = np.sort(rng.integers(20, 900, size=steps))[::-1]
        for name, off in zip(SESSION_FLOW[:steps], offsets):
            meta = {"platform": platform, "app_version": version, "screen": name}
            if name == "search":
                meta["query"] = str(rng.choice(SEARCH_TERMS))
            elif name in ("view_product", "add_to_cart"):
                meta["product_id"] = str(product_ids[min(np.searchsorted(pop_cum, rng.random() * pop_cum[-1]), len(pop) - 1)])
            rows.append((customer, f"SES{session:07d}", name, end_ts - np.timedelta64(int(off), "s"),
                         json.dumps(meta, sort_keys=True)))
    rows = sorted(rows[:target], key=lambda r: (r[3], r[1]))
    return pd.DataFrame({
        "event_id": make_ids("EVT", len(rows), 7),
        "customer_id": [r[0] for r in rows],
        "session_id": [r[1] for r in rows],
        "event_name": [r[2] for r in rows],
        "event_ts": np.array([r[3] for r in rows], dtype="datetime64[s]"),
        "metadata": [r[4] for r in rows],
    })


def build_application_logs(ctx: GenContext, ref: ReferenceData, tx: Transactions, plan: AnomalyPlan) -> pd.DataFrame:
    """Service logs; payment-gateway failure windows produce bursts of payment-service ERRORs."""
    rng = ctx.rng("events", "logs")
    target = ctx.targets["application_logs"]
    order_ids = tx.orders["order_id"].to_numpy()
    product_ids = ref.products["product_id"].to_numpy()
    store_ids = ref.stores["store_id"].to_numpy()

    def message(level: str) -> str:
        template = str(rng.choice(LOG_TEMPLATES[level]))
        return template.format(order=rng.choice(order_ids), n=int(rng.integers(1, 9)),
                               amt=f"{rng.uniform(50, 2500):.2f}", eta=int(rng.integers(5, 40)),
                               term=rng.choice(SEARCH_TERMS), product=rng.choice(product_ids),
                               store=rng.choice(store_ids), lat=int(rng.integers(800, 6000)),
                               ratio=f"{rng.uniform(0.6, 0.99):.2f}", flag=str(rng.random() < 0.5).lower())

    rows: list[tuple] = []
    burst_per_window = max(1, min(40, target // 50))
    for f in plan.gateway_failures:
        span = int((f.end_utc - f.start_utc) // np.timedelta64(1, "s"))
        for sec in np.sort(rng.integers(0, span, size=burst_per_window)):
            rows.append((f.start_utc + np.timedelta64(int(sec), "s"), "payment-service", "ERROR",
                         f"Payment declined for order {rng.choice(order_ids)}, gateway=\"pg-sim\", code=503"))
    n_background = target - len(rows)
    days = rng.integers(0, ctx.n_days, size=n_background)
    hours = rng.choice(24, size=n_background, p=(HOUR_W_DAY + 0.5) / (HOUR_W_DAY + 0.5).sum())
    levels = rng.choice(list(LEVEL_P), size=n_background, p=list(LEVEL_P.values()))
    services = rng.choice(SERVICES, size=n_background)
    for d, h, lvl, svc in zip(days, hours, levels, services):
        ts = ctx.day_start_utc(int(d)) + np.timedelta64(int(h) * 3600 + int(rng.integers(3600)), "s")
        rows.append((ts, str(svc), str(lvl), message(str(lvl))))
    rows.sort(key=lambda r: r[0])
    return pd.DataFrame({
        "log_id": make_ids("LOG", len(rows), 6),
        "event_ts": np.array([r[0] for r in rows], dtype="datetime64[s]"),
        "service": [r[1] for r in rows],
        "log_level": [r[2] for r in rows],
        "message": [r[3] for r in rows],
    })
