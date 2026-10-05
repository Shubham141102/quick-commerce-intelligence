"""Transactions that follow from orders, generated consistently (rules C1-C11, Project_Plan_v2.md §3.7).

Order items, order promotions and totals, payments (with failed retries),
deliveries, cancellations, refunds and reviews.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generation.anomalies import AnomalyPlan
from src.generation.context import GenContext, make_ids
from src.generation.customers import CustomerData
from src.generation.inventory import InventoryResult
from src.generation.orders import OrderBook
from src.generation.reference import ReferenceData
from src.generation.weather import WeatherData

MINUTE = np.timedelta64(60, "s")
PAYMENT_METHODS = {"upi": 0.55, "card": 0.20, "wallet": 0.10, "cod": 0.15}
PRE_DISPATCH_REASONS = ["customer_changed_mind", "item_unavailable", "payment_issue", "duplicate_order"]
POST_DISPATCH_REASONS = ["delivery_delayed", "customer_unreachable", "address_issue"]
REFUND_REASONS = ["damaged_item", "missing_item", "quality_issue", "expired_product"]
RATING_P_NORMAL = [0.05, 0.07, 0.13, 0.35, 0.40]
RATING_P_LATE = [0.15, 0.15, 0.25, 0.25, 0.20]
COMMENTS = {
    "good": ["Fresh and well packed, delivered quickly.", "Great quality, will order again!",
             'Arrived in "10 minutes" flat, amazing.', "Good value for money, packaging was neat.",
             "Exactly as described. Thanks!"],
    "neutral": ["Okay product, nothing special.", "Delivery was fine, product average.",
                "Decent, but a bit pricey.", "Packaging could be better, product is fine."],
    "bad": ["Item was damaged, not happy.", "Delivered late, and the product was stale.",
            "Wrong item received, had to raise a complaint.", "Poor quality, would not recommend."],
}


@dataclass
class Transactions:
    orders: pd.DataFrame
    order_items: pd.DataFrame
    order_promotions: pd.DataFrame
    payments: pd.DataFrame
    deliveries: pd.DataFrame
    cancellations: pd.DataFrame
    returns_refunds: pd.DataFrame
    reviews: pd.DataFrame


def build_transactions(ctx: GenContext, ref: ReferenceData, cust: CustomerData, weather: WeatherData,
                       plan: AnomalyPlan, book: OrderBook, inv: InventoryResult) -> Transactions:
    rng = ctx.rng("downstream")
    orders = book.orders.copy()
    n = len(orders)

    # Order items
    sizes = np.array([len(b) for b in inv.baskets])
    order_idx = np.repeat(np.arange(n), sizes)
    product = np.concatenate([np.array(b, dtype=int) for b in inv.baskets])
    qty = np.concatenate([np.array(q, dtype=int) for q in inv.quantities])
    price = ref.products["price"].to_numpy()[product]
    line_amount = np.round(qty * price, 2)
    items = pd.DataFrame({
        "order_item_id": make_ids("OI", len(product), 8),
        "order_id": orders["order_id"].to_numpy()[order_idx],
        "product_id": ref.products["product_id"].to_numpy()[product],
        "quantity": qty,
        "unit_price": price,
        "order_idx": order_idx,
        "category_idx": ref.products["category_idx"].to_numpy()[product],
        "line_amount": line_amount,
    })
    gross = np.bincount(order_idx, weights=line_amount, minlength=n)

    # Order promotions: one promotion per order at most, best discount among eligible ones
    order_promotions, discount = _order_promotions(ctx, rng, ref, cust, orders, items)
    orders["gross_amount"] = np.round(gross, 2)
    orders["discount_amount"] = discount
    orders["total_amount"] = np.round(gross - discount, 2)
    stage = orders["cancel_stage"].to_numpy()
    orders["status"] = np.where(stage == "", "delivered", "cancelled")

    deliveries, delivered_ts, pickup_ts = _deliveries(ctx, rng, ref, weather, orders)
    cancellations, cancelled_ts = _cancellations(rng, orders, pickup_ts)
    payments = _payments(ctx, rng, plan, orders, delivered_ts, cancelled_ts)
    refunds = _refunds(ctx, rng, orders, payments, delivered_ts, cancelled_ts, items)
    reviews = _reviews(ctx, rng, orders, items, delivered_ts)
    return Transactions(orders, items, order_promotions, payments, deliveries, cancellations, refunds, reviews)


def _order_promotions(ctx, rng, ref, cust, orders, items):
    n = len(orders)
    promos = ref.promotions
    p_start = ((promos["start_date"].to_numpy() - ctx.days[0]) // np.timedelta64(1, "D")).astype(int)
    p_end = ((promos["end_date"].to_numpy() - ctx.days[0]) // np.timedelta64(1, "D")).astype(int)
    p_cat = promos["category_idx"].to_numpy()
    p_disc = promos["discount_pct"].to_numpy()

    # Per order: categories present and amount per category
    by_order = items.groupby(["order_idx", "category_idx"], sort=True)["line_amount"].sum()
    cat_amount: dict[int, dict[int, float]] = {}
    for (oi, ci), amt in by_order.items():
        cat_amount.setdefault(int(oi), {})[int(ci)] = float(amt)

    day = orders["day_idx"].to_numpy()
    best_promo = np.full(n, -1)
    for i in range(n):
        cats = cat_amount.get(i, {})
        live = np.flatnonzero((p_start <= day[i]) & (p_end >= day[i]) & np.isin(p_cat, list(cats)))
        if len(live):
            best_promo[i] = live[np.argmax(p_disc[live])]

    eligible = np.flatnonzero(best_promo >= 0)
    sens = np.array([ctx.cfg.personas[p].promo_sensitivity for p in cust.persona_names])
    w = sens[orders["persona_idx"].to_numpy()[eligible]] + 0.1
    k = min(ctx.targets["order_promotions"], len(eligible))
    chosen = np.sort(rng.choice(eligible, size=k, replace=False, p=w / w.sum()))

    discount = np.zeros(n)
    for i in chosen:
        pr = best_promo[i]
        discount[i] = round(cat_amount[int(i)][int(p_cat[pr])] * p_disc[pr] / 100, 2)
    frame = pd.DataFrame({
        "order_id": orders["order_id"].to_numpy()[chosen],
        "promotion_id": promos["promotion_id"].to_numpy()[best_promo[chosen]],
        "discount_amount": discount[chosen],
        "order_idx": chosen,
    })
    return frame, discount


def _deliveries(ctx, rng, ref, weather, orders):
    cfg = ctx.cfg
    n = len(orders)
    stage = orders["cancel_stage"].to_numpy()
    ships = np.flatnonzero(stage != "pre_dispatch")
    store = orders["store_idx"].to_numpy()
    order_ts = orders["order_ts"].to_numpy().astype("datetime64[s]")

    partners = ref.partners
    active = partners["availability_status"].to_numpy() == "active"
    by_store = [np.flatnonzero(active & (partners["store_idx"].to_numpy() == s)) for s in range(len(ref.stores))]
    partner_idx = np.array([by_store[store[i]][rng.integers(len(by_store[store[i]]))] for i in ships])

    prep = rng.lognormal(np.log(4.0), 0.4, size=len(ships))
    med = cfg.demand.delivery_minutes_lognormal
    rain = weather.daily_rain_mm[ref.stores["city_idx"].to_numpy()[store[ships]], orders["day_idx"].to_numpy()[ships]]
    travel = rng.lognormal(np.log(med["median"]), med["sigma"], size=len(ships)) * (1 + 0.02 * rain)
    pickup = order_ts[ships] + (prep * 60).astype("timedelta64[s]")
    delivered = pickup + (travel * 60).astype("timedelta64[s]")

    post = stage[ships] == "post_dispatch"
    status = np.where(post, np.where(rng.random(len(ships)) < 0.6, "cancelled", "failed"), "delivered")
    delivered_out = delivered.copy()
    delivered_out[post] = np.datetime64("NaT")

    frame = pd.DataFrame({
        "delivery_id": make_ids("DEL", len(ships), 7),
        "order_id": orders["order_id"].to_numpy()[ships],
        "partner_id": partners["partner_id"].to_numpy()[partner_idx],
        "pickup_ts": pickup,
        "delivered_ts": delivered_out,
        "status": status,
        "order_idx": ships,
    })
    pickup_all = np.full(n, np.datetime64("NaT"), dtype="datetime64[s]")
    delivered_all = pickup_all.copy()
    pickup_all[ships] = pickup
    delivered_all[ships] = delivered_out
    return frame, delivered_all, pickup_all


def _cancellations(rng, orders, pickup_ts):
    n = len(orders)
    stage = orders["cancel_stage"].to_numpy()
    idx = np.flatnonzero(stage != "")
    order_ts = orders["order_ts"].to_numpy().astype("datetime64[s]")
    is_post = stage[idx] == "post_dispatch"
    ts = np.where(
        is_post,
        pickup_ts[idx] + rng.integers(3 * 60, 20 * 60, size=len(idx)).astype("timedelta64[s]"),
        order_ts[idx] + rng.integers(60, 6 * 60, size=len(idx)).astype("timedelta64[s]"),
    )
    reason = np.where(is_post, rng.choice(POST_DISPATCH_REASONS, size=len(idx)),
                      rng.choice(PRE_DISPATCH_REASONS, size=len(idx)))
    frame = pd.DataFrame({
        "cancellation_id": make_ids("CAN", len(idx), 6),
        "order_id": orders["order_id"].to_numpy()[idx],
        "stage": stage[idx],
        "reason": reason,
        "cancelled_ts": ts,
        "order_idx": idx,
    })
    cancelled_all = np.full(n, np.datetime64("NaT"), dtype="datetime64[s]")
    cancelled_all[idx] = ts
    return frame, cancelled_all


def _payments(ctx, rng, plan, orders, delivered_ts, cancelled_ts):
    n = len(orders)
    order_ts = orders["order_ts"].to_numpy().astype("datetime64[s]")
    stage = orders["cancel_stage"].to_numpy()
    method = rng.choice(list(PAYMENT_METHODS), size=n, p=list(PAYMENT_METHODS.values()))

    # Failed first attempts (prepaid only), concentrated in gateway-failure windows
    prepaid = np.flatnonzero(method != "cod")
    w = np.ones(len(prepaid))
    for f in plan.gateway_failures:
        w[(order_ts[prepaid] >= f.start_utc) & (order_ts[prepaid] < f.end_utc)] *= 15
    n_retry = min(ctx.targets["payments"] - n, len(prepaid))
    retry = np.zeros(n, dtype=bool)
    retry[rng.choice(prepaid, size=n_retry, replace=False, p=w / w.sum())] = True

    total = orders["total_amount"].to_numpy()
    rows = []
    for i in range(n):
        if retry[i]:
            rows.append((i, 1, method[i], total[i], "failed", order_ts[i] + np.timedelta64(int(rng.integers(5, 40)), "s")))
        attempt = 2 if retry[i] else 1
        if method[i] == "cod":
            if stage[i] == "":
                status, ts = "success", delivered_ts[i]
            else:
                status, ts = "cancelled", cancelled_ts[i]
        else:
            ts = order_ts[i] + np.timedelta64(int(rng.integers(45, 180)), "s")
            status = "voided" if stage[i] == "pre_dispatch" else "success"
        rows.append((i, attempt, method[i], total[i], status, ts))

    return pd.DataFrame({
        "payment_id": make_ids("PAY", len(rows), 7),
        "order_id": orders["order_id"].to_numpy()[[r[0] for r in rows]],
        "attempt_no": [r[1] for r in rows],
        "method": [r[2] for r in rows],
        "amount": [r[3] for r in rows],
        "status": [r[4] for r in rows],
        "payment_ts": np.array([r[5] for r in rows], dtype="datetime64[s]"),
        "order_idx": [r[0] for r in rows],
    })


def _refunds(ctx, rng, orders, payments, delivered_ts, cancelled_ts, items):
    target = ctx.targets["returns_refunds"]
    end = ctx.end_utc
    success = payments[payments["status"] == "success"].set_index("order_idx")
    stage = orders["cancel_stage"].to_numpy()
    total = orders["total_amount"].to_numpy()

    post_prepaid = np.array([i for i in np.flatnonzero(stage == "post_dispatch")
                             if i in success.index and success.at[i, "method"] != "cod"], dtype=int)
    if len(post_prepaid) > target:
        post_prepaid = np.sort(rng.choice(post_prepaid, size=target, replace=False))
    delivered = np.array([i for i in np.flatnonzero(stage == "")
                          if i in success.index and delivered_ts[i] <= end - np.timedelta64(3, "D")], dtype=int)
    partial = np.sort(rng.choice(delivered, size=min(target - len(post_prepaid), len(delivered)), replace=False))

    lines = items.groupby("order_idx")["line_amount"].apply(list)
    rows = []
    for i in post_prepaid:
        ts = min(cancelled_ts[i] + np.timedelta64(int(rng.integers(1, 49)) * 3600, "s"), end)
        rows.append((i, total[i], "order_cancelled", ts))
    for i in partial:
        amount = min(float(rng.choice(lines[i])), float(total[i]))
        ts = min(delivered_ts[i] + np.timedelta64(int(rng.integers(2, 73)) * 3600, "s"), end)
        rows.append((i, amount, str(rng.choice(REFUND_REASONS)), ts))
    rows.sort(key=lambda r: r[3])
    idx = [r[0] for r in rows]
    return pd.DataFrame({
        "refund_id": make_ids("REF", len(rows), 6),
        "payment_id": success.loc[idx, "payment_id"].to_numpy(),
        "order_id": orders["order_id"].to_numpy()[idx],
        "amount": np.round([r[1] for r in rows], 2),
        "reason": [r[2] for r in rows],
        "status": rng.choice(["completed", "pending", "rejected"], size=len(rows), p=[0.90, 0.07, 0.03]),
        "refund_ts": np.array([r[3] for r in rows], dtype="datetime64[s]"),
        "order_idx": idx,
    })


def _reviews(ctx, rng, orders, items, delivered_ts):
    cfg = ctx.cfg
    end = ctx.end_utc
    order_ts = orders["order_ts"].to_numpy().astype("datetime64[s]")
    ok = np.flatnonzero((orders["cancel_stage"].to_numpy() == "") & (delivered_ts <= end - np.timedelta64(1, "D")))
    chosen = np.sort(rng.choice(ok, size=min(ctx.targets["reviews"], len(ok)), replace=False))
    products = items.groupby("order_idx")["product_id"].apply(list)
    late_cut = np.timedelta64(cfg.demand.delivery_sla_minutes + 10, "m")
    no_comment = cfg.dirty["outside_budget"]["missing_optional"]["reviews.comment"]

    rows = []
    for i in chosen:
        late = (delivered_ts[i] - order_ts[i]) > late_cut
        rating = int(rng.choice([1, 2, 3, 4, 5], p=RATING_P_LATE if late else RATING_P_NORMAL))
        mood = "good" if rating >= 4 else "neutral" if rating == 3 else "bad"
        comment = None if rng.random() < no_comment else str(rng.choice(COMMENTS[mood]))
        ts = min(delivered_ts[i] + np.timedelta64(int(rng.integers(1, 73)) * 3600, "s"), end)
        rows.append((i, str(rng.choice(products[i])), rating, comment, ts))
    rows.sort(key=lambda r: r[4])
    idx = [r[0] for r in rows]
    return pd.DataFrame({
        "review_id": make_ids("REV", len(rows), 6),
        "customer_id": orders["customer_id"].to_numpy()[idx],
        "product_id": [r[1] for r in rows],
        "order_id": orders["order_id"].to_numpy()[idx],
        "rating": [r[2] for r in rows],
        "comment": [r[3] for r in rows],
        "review_ts": np.array([r[4] for r in rows], dtype="datetime64[s]"),
        "order_idx": idx,
    })
