"""Orders and baskets from the demand model (Project_Plan_v2.md §3.6).

1. Allocate the exact order count across store-days by weight
   (store size x day-of-week x rain x trend, adjusted for planted anomalies).
2. Pick a customer for each order (same city 97%, signed up before the order).
3. Pick an order time from the persona's hour profile.
4. Size baskets per persona, then nudge sizes so the line total hits the target exactly.
5. Fill baskets: category by persona mix (+ promotion boost), product by Zipf within category.
6. Apply planted spikes and affinity pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import WEEKDAYS
from src.generation.anomalies import AnomalyPlan
from src.generation.context import IST_OFFSET, GenContext, make_ids
from src.generation.customers import CustomerData
from src.generation.reference import ReferenceData, active_promotions_matrix
from src.generation.weather import WeatherData

HOUR_W_DAY = np.array([0, 0, 0, 0, 0, 0, 2, 5, 7, 7, 6, 5, 5, 5, 4, 4, 5, 6, 8, 9, 9, 7, 4, 2], dtype=float)
HOUR_W_NIGHT = np.zeros(24)
HOUR_W_NIGHT[[21, 22, 23, 0, 1]] = [3, 4, 4, 3, 2]

PERSONA_CATEGORY_BOOST = {
    "daily_essentials": ["Dairy & Eggs", "Bakery", "Fruits", "Vegetables", "Atta, Rice & Dal", "Tea & Coffee"],
    "weekend_stock_up": ["Atta, Rice & Dal", "Household Cleaning", "Personal Care", "Breakfast & Cereals",
                         "Baby Care", "Masala & Spices", "Oils", "Kitchen & Home"],
    "snacks_at_night": ["Snacks & Namkeen", "Beverages", "Ice Cream & Desserts", "Instant & Frozen Food",
                        "Sweets & Chocolates"],
    "premium": ["Gourmet & Organic", "Meat & Seafood", "Health & Wellness", "Personal Care", "Tea & Coffee"],
    "deal_seeker": [],
}
QUANTITY_P = {1: 0.70, 2: 0.22, 3: 0.06, 4: 0.015, 5: 0.005}
MAX_BASKET = 15
OTHER_CITY_SHARE = 0.03
SPIKE_ATTACH = 0.25


@dataclass
class OrderBook:
    orders: pd.DataFrame          # one row per order, sorted by order_ts (UTC)
    baskets: list[list[int]]      # product indices per order
    quantities: list[list[int]]


def weekday_index(days: np.ndarray) -> np.ndarray:
    """Monday=0 ... Sunday=6 for datetime64[D] values (1970-01-01 was a Thursday)."""
    return (days.astype(int) + 3) % 7


def _store_day_weights(ctx: GenContext, ref: ReferenceData, weather: WeatherData, plan: AnomalyPlan) -> np.ndarray:
    d = ctx.cfg.demand
    dow = np.array([d.day_of_week_factors[w] for w in WEEKDAYS])[weekday_index(ctx.days)]
    rain = weather.daily_rain_mm[ref.stores["city_idx"].to_numpy()]
    trend = (1 + d.monthly_trend) ** (np.arange(ctx.n_days) / 30.44)
    w = ref.stores["demand_factor"].to_numpy()[:, None] * dow * (1 + d.rain_uplift_per_10mm * rain / 10) * trend
    for o in plan.outages:
        w[o.store_idx, o.day_idx] *= 1 - HOUR_W_DAY[o.start_hour:o.end_hour].sum() / HOUR_W_DAY.sum()
    for s in plan.spikes:
        w[s.store_idx, s.day_from:s.day_to + 1] *= 1.3
    return w


def _hour_cdfs(blocked: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    day, night = HOUR_W_DAY.copy(), HOUR_W_NIGHT.copy()
    if blocked:
        day[blocked[0]:blocked[1]] = 0
        night[blocked[0]:blocked[1]] = 0
    return np.cumsum(day) / day.sum(), np.cumsum(night) / night.sum()


def build_order_skeleton(ctx: GenContext, ref: ReferenceData, cust: CustomerData,
                         weather: WeatherData, plan: AnomalyPlan) -> pd.DataFrame:
    cfg = ctx.cfg
    rng = ctx.rng("orders", "skeleton")
    n_orders = ctx.targets["orders"]
    n_stores, n_days = len(ref.stores), ctx.n_days

    w = _store_day_weights(ctx, ref, weather, plan)
    counts = rng.multinomial(n_orders, (w / w.sum()).ravel()).reshape(n_stores, n_days)

    c = cust.customers
    persona = c["persona_idx"].to_numpy()
    city = c["city_idx"].to_numpy()
    signup_off = ((c["signup_date"].to_numpy().astype("datetime64[D]") - ctx.days[0])
                  // np.timedelta64(1, "D")).astype(int)
    pday = np.ones((len(cust.persona_names), 7))
    if "weekend_stock_up" in cust.persona_names:
        k = cust.persona_names.index("weekend_stock_up")
        pday[k] = [0.7, 0.7, 0.7, 0.7, 0.7, 2.0, 2.0]
    night_share = np.array([cfg.personas[p].night_share for p in cust.persona_names])

    # Per city: customers sorted by signup, cumulative weights per weekday
    by_city = []
    for ci in range(cfg.counts.cities):
        idx = np.flatnonzero(city == ci)
        idx = idx[np.argsort(signup_off[idx], kind="stable")]
        cum = np.cumsum(c["activity"].to_numpy()[idx][None, :] * pday[persona[idx]].T, axis=1)  # [7, n]
        by_city.append((idx, signup_off[idx], cum))

    def draw_customers(ci: int, day: int, wd: int, k: int) -> np.ndarray:
        idx, signups, cum = by_city[ci]
        m = int(np.searchsorted(signups, day, side="right")) or len(idx)
        r = rng.random(k) * cum[wd, m - 1]
        return idx[np.minimum(np.searchsorted(cum[wd, :m], r), m - 1)]

    outage_map = {(o.store_idx, o.day_idx): (o.start_hour, o.end_hour) for o in plan.outages}
    default_cdfs = _hour_cdfs(None)
    weekdays = weekday_index(ctx.days)
    store_city = ref.stores["city_idx"].to_numpy()

    store_col, day_col, cust_col, ts_col = [], [], [], []
    for s in range(n_stores):
        for d in range(n_days):
            k = int(counts[s, d])
            if k == 0:
                continue
            ci, wd = int(store_city[s]), int(weekdays[d])
            other = rng.random(k) < OTHER_CITY_SHARE
            chosen = draw_customers(ci, d, wd, k)
            if other.any() and cfg.counts.cities > 1:
                for j in np.flatnonzero(other):
                    oc = int(rng.choice([x for x in range(cfg.counts.cities) if x != ci]))
                    chosen[j] = draw_customers(oc, d, wd, 1)[0]
            blocked = outage_map.get((s, d))
            day_cdf, night_cdf = _hour_cdfs(blocked) if blocked else default_cdfs
            is_night = rng.random(k) < night_share[persona[chosen]]
            hours = np.where(is_night, np.searchsorted(night_cdf, rng.random(k), side="right"),
                             np.searchsorted(day_cdf, rng.random(k), side="right"))
            secs = hours * 3600 + rng.integers(0, 3600, size=k)
            store_col.append(np.full(k, s))
            day_col.append(np.full(k, d))
            cust_col.append(chosen)
            ts_col.append(ctx.days[d].astype("datetime64[s]") + secs.astype("timedelta64[s]") - IST_OFFSET)

    orders = pd.DataFrame({
        "store_idx": np.concatenate(store_col),
        "day_idx": np.concatenate(day_col),
        "cust_idx": np.concatenate(cust_col),
        "order_ts": np.concatenate(ts_col),
    })
    orders = orders.sort_values(["order_ts", "store_idx"], kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", make_ids("ORD", len(orders), 7))
    orders["customer_id"] = c["customer_id"].to_numpy()[orders["cust_idx"].to_numpy()]
    orders["store_id"] = ref.stores["store_id"].to_numpy()[orders["store_idx"].to_numpy()]
    orders["persona_idx"] = persona[orders["cust_idx"].to_numpy()]

    # Outlier (very large but valid) orders
    n_out = round(n_orders * cfg.dirty["outside_budget"]["outliers"]["orders.large_order"])
    orders["is_outlier"] = False
    orders.loc[rng.choice(n_orders, size=n_out, replace=False), "is_outlier"] = True

    # Cancellation stage, decided up front (pre-dispatch orders never deplete stock).
    # Post-dispatch cancellations are twice as likely on heavy-rain days.
    n_cancel = ctx.targets["cancellations"]
    n_pre = n_orders - ctx.targets["deliveries"]
    rain = weather.daily_rain_mm[store_city[orders["store_idx"].to_numpy()], orders["day_idx"].to_numpy()]
    w_post = np.where(rain > 10, 2.0, 1.0)
    post = rng.choice(n_orders, size=n_cancel - n_pre, replace=False, p=w_post / w_post.sum())
    remaining = np.setdiff1d(np.arange(n_orders), post)
    pre = rng.choice(remaining, size=n_pre, replace=False)
    stage = np.full(n_orders, "", dtype=object)
    stage[post] = "post_dispatch"
    stage[pre] = "pre_dispatch"
    orders["cancel_stage"] = stage
    return orders


def _hit_total(sizes: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    sizes = sizes.copy()
    for _ in range(100):
        diff = target - int(sizes.sum())
        if diff == 0:
            return sizes
        step = 1 if diff > 0 else -1
        cand = np.flatnonzero(sizes < MAX_BASKET) if step > 0 else np.flatnonzero(sizes > 1)
        pick = rng.choice(cand, size=min(abs(diff), len(cand)), replace=False)
        sizes[pick] += step
    raise RuntimeError("could not reach the order-item target")


def _persona_category_weights(ctx: GenContext, ref: ReferenceData, persona_names: list[str]) -> np.ndarray:
    names = ref.categories["category_name"].tolist()
    w = np.ones((len(persona_names), len(names)))
    for p, persona in enumerate(persona_names):
        for cat in PERSONA_CATEGORY_BOOST.get(persona, []):
            if cat in names:
                w[p, names.index(cat)] = 6.0
    return w


def fill_baskets(ctx: GenContext, ref: ReferenceData, cust: CustomerData,
                 orders: pd.DataFrame, plan: AnomalyPlan) -> OrderBook:
    cfg = ctx.cfg
    rng = ctx.rng("orders", "baskets")
    n_orders = len(orders)
    persona = orders["persona_idx"].to_numpy()
    day = orders["day_idx"].to_numpy()

    avg = np.array([cfg.personas[p].avg_basket for p in cust.persona_names])[persona]
    sizes = np.clip(1 + rng.poisson(np.maximum(avg - 1, 0)), 1, MAX_BASKET)
    sizes = _hit_total(sizes, ctx.targets["order_items"], rng)

    # Category weights per persona and day (active promotions boost their category)
    promo_sens = np.array([cfg.personas[p].promo_sensitivity for p in cust.persona_names])
    base = _persona_category_weights(ctx, ref, cust.persona_names)
    active = active_promotions_matrix(ctx, ref)
    cat_cum = np.cumsum(base[:, None, :] * (1 + active[None, :, :] * (0.3 + promo_sens[:, None, None])), axis=2)

    prod_cat = ref.products["category_idx"].to_numpy()
    pop = ref.products["popularity_weight"].to_numpy()
    products_by_cat = [np.flatnonzero(prod_cat == ci) for ci in range(len(ref.categories))]
    prod_cum = [np.cumsum(pop[idx]) for idx in products_by_cat]

    item_order = np.repeat(np.arange(n_orders), sizes)
    n_items = len(item_order)

    def draw(items: np.ndarray) -> np.ndarray:
        out = np.empty(len(items), dtype=int)
        for lo in range(0, len(items), 200_000):
            chunk = items[lo:lo + 200_000]
            cum = cat_cum[persona[item_order[chunk]], day[item_order[chunk]]]
            cats = (cum < (rng.random(len(chunk)) * cum[:, -1])[:, None]).sum(axis=1)
            for ci in np.unique(cats):
                sel = np.flatnonzero(cats == ci)
                pos = np.searchsorted(prod_cum[ci], rng.random(len(sel)) * prod_cum[ci][-1])
                out[lo + sel] = products_by_cat[ci][np.minimum(pos, len(products_by_cat[ci]) - 1)]
        return out

    product = draw(np.arange(n_items))
    for _ in range(30):  # products must be distinct within an order
        key = item_order.astype(np.int64) * ref.n_products + product
        _, first = np.unique(key, return_index=True)
        dup = np.setdiff1d(np.arange(n_items), first)
        if len(dup) == 0:
            break
        product[dup] = draw(dup)

    offsets = np.concatenate([[0], np.cumsum(sizes)])
    baskets = [list(dict.fromkeys(product[offsets[i]:offsets[i + 1]].tolist())) for i in range(n_orders)]
    for i, b in enumerate(baskets):  # rare leftovers: top up with unused products
        while len(b) < sizes[i]:
            p = int(rng.integers(ref.n_products))
            if p not in b:
                b.append(p)

    _apply_spikes(rng, orders, baskets, plan)
    _apply_affinity(ctx, rng, ref, baskets)

    q_vals, q_p = np.array(list(QUANTITY_P)), np.array(list(QUANTITY_P.values()))
    outlier = orders["is_outlier"].to_numpy()
    quantities = []
    for i, b in enumerate(baskets):
        if outlier[i]:
            quantities.append(rng.integers(8, 21, size=len(b)).tolist())
        else:
            quantities.append(rng.choice(q_vals, size=len(b), p=q_p).tolist())
    return OrderBook(orders, baskets, quantities)


def _apply_spikes(rng: np.random.Generator, orders: pd.DataFrame, baskets: list[list[int]], plan: AnomalyPlan) -> None:
    store, day = orders["store_idx"].to_numpy(), orders["day_idx"].to_numpy()
    for s in plan.spikes:
        for i in np.flatnonzero((store == s.store_idx) & (day >= s.day_from) & (day <= s.day_to)):
            b = baskets[i]
            if s.product_idx not in b and rng.random() < SPIKE_ATTACH:
                b[int(rng.integers(len(b)))] = s.product_idx


def _apply_affinity(ctx: GenContext, rng: np.random.Generator, ref: ReferenceData, baskets: list[list[int]]) -> None:
    """Plant co-purchase pairs: when A is in a basket, B replaces another item with probability p."""
    cfg = ctx.cfg
    rank = ref.products["popularity_rank"].to_numpy()
    prod_cat = ref.products["category_idx"].to_numpy()
    pool = np.flatnonzero((rank > cfg.focus_skus) & (rank <= cfg.focus_skus + 150))
    pool = pool[np.argsort(rank[pool], kind="stable")]
    lo, hi = cfg.planted.affinity_attach_probability
    pairs: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    order = rng.permutation(pool).tolist()
    for a in order:
        if len(pairs) >= cfg.planted.affinity_pairs:
            break
        if a in used:
            continue
        partners = [b for b in order if b not in used and b != a and prod_cat[b] != prod_cat[a]]
        if not partners:
            continue
        b = partners[0]
        used.update((a, b))
        pairs[a] = (b, float(rng.uniform(lo, hi)))

    for basket in baskets:
        if len(basket) < 2:
            continue
        for a in [p for p in basket if p in pairs]:
            b, p = pairs[a]
            if b not in basket and rng.random() < p:
                slots = [j for j, x in enumerate(basket) if x != a and x not in pairs]
                if slots:
                    basket[slots[int(rng.integers(len(slots)))]] = b

    pid = ref.products["product_id"].to_numpy()
    ctx.ground_truth["gt_affinity_pairs"] = [
        {"antecedent_product_id": pid[a], "consequent_product_id": pid[b], "attach_probability": round(p, 3)}
        for a, (b, p) in sorted(pairs.items(), key=lambda kv: pid[kv[0]])
    ]
