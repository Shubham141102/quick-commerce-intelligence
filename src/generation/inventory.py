"""Inventory simulation for focus SKUs (Project_Plan_v2.md §3.6 step 5).

Stock is simulated chronologically per (store, focus SKU):
- every order line depletes stock (pre-dispatch cancellations only check availability);
- a nightly check at 22:00 IST places a restock when stock <= reorder level,
  arriving 1-2 days later at 07:00 IST and topping stock up to `order_up_to`;
- background damage / cycle-count adjustment events change stock at random times;
- planted stockout episodes write off most of the stock and block restocks for 4-7 days.

When a customer asks for an item that is out of stock, the line is filled with a
non-tracked substitute from the same category (or reduced to the available quantity).
Lost demand and every stockout interval go to ground truth.

The number of background events is tuned (re-running the simulation) so the total
event count hits the dataset target.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.generation.context import GenContext, make_ids
from src.generation.orders import OrderBook
from src.generation.reference import ReferenceData

HOUR = np.timedelta64(3600, "s")
CHECK_HOUR_IST, ARRIVAL_HOUR_IST, EPISODE_HOUR_IST = 22, 7, 8
# Event priorities when timestamps tie
P_ARRIVE, P_CHANGE, P_CHECK, P_SNAPSHOT = 0, 1, 2, 3


@dataclass
class InventoryResult:
    snapshots: pd.DataFrame
    events: pd.DataFrame
    baskets: list[list[int]]
    quantities: list[list[int]]


@dataclass
class _Episode:
    pair: int
    start: np.datetime64
    end: np.datetime64
    episode_id: str


def simulate_inventory(ctx: GenContext, ref: ReferenceData, book: OrderBook) -> InventoryResult:
    focus = ref.focus_products
    n_stores, n_focus = len(ref.stores), len(focus)
    focus_pos = {int(p): j for j, p in enumerate(focus)}

    # Average daily demand per pair (orders that will actually ship) -> reorder policy
    store_idx = book.orders["store_idx"].to_numpy()
    shipped = book.orders["cancel_stage"].to_numpy() != "pre_dispatch"
    units = np.zeros(n_stores * n_focus)
    for i, (basket, qty) in enumerate(zip(book.baskets, book.quantities)):
        if shipped[i]:
            for p, q in zip(basket, qty):
                j = focus_pos.get(p)
                if j is not None:
                    units[store_idx[i] * n_focus + j] += q
    mean_daily = units / ctx.n_days
    reorder = np.ceil(mean_daily * 3).astype(int) + 1
    order_up_to = reorder + np.ceil(mean_daily * 7).astype(int) + 2

    episodes = _plan_episodes(ctx, n_stores * n_focus)
    target = ctx.targets["inventory_events"]
    # Restock counts depend on the background events, so tune their number until the
    # simulated total lands just under the target; the last few are end-of-period
    # cycle-count adjustments (added after every snapshot, so nothing observed changes).
    n_background = round(target * 0.35)
    result = None
    for attempt in range(8):
        result, n_before_top_up = _run(ctx, ref, book, focus_pos, reorder, order_up_to,
                                       episodes, n_background, attempt, target)
        gap = target - n_before_top_up
        if 0 <= gap <= 5:
            break
        n_background = max(0, n_background + gap - 2)
    return result


def _plan_episodes(ctx: GenContext, n_pairs: int) -> list[_Episode]:
    rng = ctx.rng("inventory", "episodes")
    lo, hi = min(3, ctx.n_days // 10), max(4, ctx.n_days - 8)
    episodes = []
    for e, pair in enumerate(rng.choice(n_pairs, size=ctx.cfg.planted.stockout_episodes, replace=True)):
        d = int(rng.integers(lo, hi))
        start = ctx.day_start_utc(d) + EPISODE_HOUR_IST * HOUR
        end = ctx.day_start_utc(min(ctx.n_days - 1, d + int(rng.integers(4, 8)))) + ARRIVAL_HOUR_IST * HOUR
        episodes.append(_Episode(int(pair), start, end, f"SO{e + 1:03d}"))
    return episodes


def _run(ctx, ref, book, focus_pos, reorder, order_up_to, episodes, n_background, attempt, target):
    cfg = ctx.cfg
    rng = ctx.rng("inventory", "run", attempt)
    focus = ref.focus_products
    n_focus, n_stores = len(focus), len(ref.stores)
    n_pairs = n_stores * n_focus
    end_utc = ctx.end_utc
    lead_lo, lead_hi = cfg.demand.restock_lead_time_days

    baskets = [list(b) for b in book.baskets]
    quantities = [list(q) for q in book.quantities]
    stock = order_up_to.copy()
    pending = np.zeros(n_pairs, dtype=bool)
    out_since: dict[int, np.datetime64] = {}
    lost_open: dict[int, int] = {}
    stockouts: list[dict] = []
    events: list[tuple] = []      # (ts, pair, event_type, quantity)
    snapshots: list[tuple] = []   # (ts, pair, stock, reorder)
    windows: dict[int, list[_Episode]] = {}
    for ep in episodes:
        windows.setdefault(ep.pair, []).append(ep)

    def blocked(pair: int, ts) -> _Episode | None:
        for ep in windows.get(pair, ()):
            if ep.start <= ts < ep.end:
                return ep
        return None

    def set_stock(pair: int, value: int, ts) -> None:
        before = stock[pair]
        stock[pair] = value
        if before > 0 and value == 0:
            out_since[pair] = ts
            lost_open[pair] = 0
        elif before == 0 and value > 0 and pair in out_since:
            _close(pair, ts)

    def _close(pair: int, ts) -> None:
        start = out_since.pop(pair)
        ep = blocked(pair, start)
        stockouts.append({"pair": pair, "start_ts": start, "end_ts": ts, "lost_units": lost_open.pop(pair, 0),
                          "planted_episode_id": ep.episode_id if ep else ""})

    heap: list[tuple] = []
    counter = itertools.count()

    def push(ts, priority, kind, payload) -> None:
        heapq.heappush(heap, (ts, priority, next(counter), kind, payload))

    for d in range(ctx.n_days):
        day0 = ctx.day_start_utc(d)
        for s in range(n_stores):
            push(day0 + CHECK_HOUR_IST * HOUR, P_CHECK, "check", (s, d))
    for snap_day in cfg.snapshot_dates():
        d = int((np.datetime64(snap_day) - ctx.days[0]) // np.timedelta64(1, "D"))
        push(ctx.day_start_utc(d) + cfg.demand.snapshot_hour_ist * HOUR, P_SNAPSHOT, "snapshot", None)
    for ep in episodes:
        push(ep.start, P_CHANGE, "episode", ep)
    bg_pairs = rng.integers(0, n_pairs, size=n_background)
    bg_days = rng.integers(0, ctx.n_days, size=n_background)
    bg_secs = rng.integers(7 * 3600, 23 * 3600, size=n_background)
    bg_kind = np.where(rng.random(n_background) < 0.6, "damage", "adjustment")
    for k, d, sec, kind in zip(bg_pairs, bg_days, bg_secs, bg_kind):
        push(ctx.day_start_utc(int(d)) + np.timedelta64(int(sec), "s"), P_CHANGE, "background", (int(k), str(kind)))

    n_episode_events = 0

    def process(ts, kind, payload) -> None:
        nonlocal n_episode_events
        if kind == "check":
            s, d = payload
            for j in range(n_focus):
                pair = s * n_focus + j
                if not pending[pair] and stock[pair] <= reorder[pair] and not blocked(pair, ts):
                    lead = int(rng.integers(lead_lo, lead_hi + 1))
                    if d + lead < ctx.n_days:
                        arrival = ctx.day_start_utc(d + lead) + ARRIVAL_HOUR_IST * HOUR
                    else:
                        arrival = end_utc + HOUR  # arrives after the period; never recorded
                    push(arrival, P_ARRIVE, "arrive", pair)
                    pending[pair] = True
        elif kind == "arrive":
            pair = payload
            ep = blocked(pair, ts)
            if ep:
                push(ep.end, P_ARRIVE, "arrive", pair)
                return
            qty = max(1, int(order_up_to[pair] - stock[pair]))
            events.append((ts, pair, "restock", qty))
            set_stock(pair, stock[pair] + qty, ts)
            pending[pair] = False
        elif kind == "episode":
            pair = payload.pair
            if stock[pair] > 0:
                write_off = int(np.ceil(stock[pair] * 0.6))
                events.append((ts, pair, "damage", -write_off))
                n_episode_events += 1
                set_stock(pair, stock[pair] - write_off, ts)
        elif kind == "background":
            pair, ev = payload
            if ev == "damage":
                q = -min(int(rng.integers(1, 4)), int(stock[pair]))
            else:
                q = int(rng.choice([-1, 1])) * int(rng.integers(1, 5))
                if q < 0:
                    q = -min(-q, int(stock[pair]))
            if q == 0:
                ev, q = "adjustment", int(rng.integers(1, 4))
            events.append((ts, pair, ev, q))
            set_stock(pair, stock[pair] + q, ts)
        elif kind == "snapshot":
            for pair in range(n_pairs):
                snapshots.append((ts, pair, int(stock[pair]), int(reorder[pair])))

    # Non-focus products of each category, for substitutions
    prod_cat = ref.products["category_idx"].to_numpy()
    pop = ref.products["popularity_weight"].to_numpy()
    focus_set = set(focus_pos)
    subs_by_cat = []
    for ci in range(len(ref.categories)):
        idx = np.array([p for p in np.flatnonzero(prod_cat == ci) if p not in focus_set], dtype=int)
        subs_by_cat.append((idx, np.cumsum(pop[idx]) if len(idx) else np.array([])))

    def substitute(p: int, basket: list[int]) -> int:
        idx, cum = subs_by_cat[prod_cat[p]]
        for _ in range(10):
            if len(idx) == 0:
                break
            cand = int(idx[min(np.searchsorted(cum, rng.random() * cum[-1]), len(idx) - 1)])
            if cand not in basket:
                return cand
        while True:
            cand = int(rng.integers(ref.n_products))
            if cand not in focus_set and cand not in basket:
                return cand

    order_ts = book.orders["order_ts"].to_numpy()
    store_idx = book.orders["store_idx"].to_numpy()
    pre = book.orders["cancel_stage"].to_numpy() == "pre_dispatch"
    for i in range(len(baskets)):
        ts = order_ts[i]
        while heap and heap[0][0] <= ts:
            t, _, _, kind, payload = heapq.heappop(heap)
            process(t, kind, payload)
        basket, qty = baskets[i], quantities[i]
        for line, p in enumerate(basket):
            j = focus_pos.get(p)
            if j is None:
                continue
            pair = store_idx[i] * n_focus + j
            available = int(stock[pair])
            if available == 0:
                basket[line] = substitute(p, basket)
                lost = qty[line]
            elif qty[line] > available:
                lost = qty[line] - available
                qty[line] = available
            else:
                lost = 0
            if lost and pair in lost_open:
                lost_open[pair] += lost
            if not pre[i] and basket[line] == p:
                set_stock(pair, available - qty[line], ts)
    while heap and heap[0][0] <= end_utc:
        t, _, _, kind, payload = heapq.heappop(heap)
        process(t, kind, payload)
    n_before_top_up = len(events)
    top_up_ts = end_utc - np.timedelta64(1800, "s")  # 23:29:59 IST on the last day
    for k in range(max(0, target - n_before_top_up)):
        pair = int(rng.integers(n_pairs))
        q = int(rng.integers(1, 4))
        ts = top_up_ts + np.timedelta64(k, "s")
        events.append((ts, pair, "adjustment", q))
        set_stock(pair, stock[pair] + q, ts)
    for pair in list(out_since):
        _close(pair, end_utc)

    store_ids = ref.stores["store_id"].to_numpy()
    product_ids = ref.products["product_id"].to_numpy()

    def ids(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return store_ids[pairs // n_focus], product_ids[focus[pairs % n_focus]]

    events.sort(key=lambda e: (e[0], e[1]))
    ev_pairs = np.array([e[1] for e in events], dtype=int)
    ev_store, ev_prod = ids(ev_pairs)
    events_df = pd.DataFrame({
        "event_id": make_ids("IEV", len(events), 6),
        "store_id": ev_store, "product_id": ev_prod,
        "event_type": [e[2] for e in events],
        "quantity": [e[3] for e in events],
        "event_ts": np.array([e[0] for e in events], dtype="datetime64[s]"),
    })
    snapshots.sort(key=lambda r: (r[0], r[1]))
    sn_pairs = np.array([r[1] for r in snapshots], dtype=int)
    sn_store, sn_prod = ids(sn_pairs)
    snapshots_df = pd.DataFrame({
        "snapshot_id": make_ids("SNP", len(snapshots), 6),
        "store_id": sn_store, "product_id": sn_prod,
        "stock_quantity": [r[2] for r in snapshots],
        "reorder_level": [r[3] for r in snapshots],
        "snapshot_ts": np.array([r[0] for r in snapshots], dtype="datetime64[s]"),
    })

    gt = []
    for so in sorted(stockouts, key=lambda r: (r["start_ts"], r["pair"])):
        st, pr = ids(np.array([so["pair"]]))
        gt.append({"store_id": st[0], "product_id": pr[0], "start_ts": so["start_ts"], "end_ts": so["end_ts"],
                   "lost_units": so["lost_units"], "planted_episode_id": so["planted_episode_id"]})
    ctx.ground_truth["gt_stockouts"] = gt
    ep_rows = []
    for ep in episodes:
        st, pr = ids(np.array([ep.pair]))
        ep_rows.append({"episode_id": ep.episode_id, "store_id": st[0], "product_id": pr[0],
                        "start_ts": ep.start, "end_ts": ep.end})
    ctx.ground_truth["gt_stockout_episodes"] = ep_rows

    return InventoryResult(snapshots_df, events_df, baskets, quantities), n_before_top_up
