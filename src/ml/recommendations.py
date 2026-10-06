"""Product recommendations (Phase 5D): "buy again" + "often bought with" + popularity.

score(p) = w_repeat × customer's own purchase frequency of p (normalised to the customer's top product)
         + w_copurchase × co-purchase strength with the customer's products (normalised)
         + w_popular × global popularity (normalised)

Weights are chosen on a validation month, never on the test month:
1. train on completed orders Apr–Jul, score every weight combination of WEIGHT_GRID on August, keep the best;
2. refit on Apr–Aug and score September once (Precision@10, Recall@10, hit rate, coverage), next to a
   popularity baseline and a pure "buy again" baseline;
3. refit on all data for the recommendations shown in the app.

Revision (2026-10-05, after the first measurement): the first version used fixed weights 1.0 / 0.6 / 0.2,
assuming customers mostly re-buy the same items. In this data only ~27% of September purchases were repeats,
and the fixed hybrid lost to popularity (13.4% vs 14.0%). The grid and the August validation were set before
re-measuring. See docs/10_marketing_analytics.md.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

TOP_K = 10
N_POPULAR, N_COPURCHASE, PARTNERS_PER_PRODUCT = 30, 50, 30
DEFAULT_WEIGHTS = {"repeat": 1.0, "copurchase": 0.6, "popular": 0.2}   # first version; used only without validation data
WEIGHT_GRID = {"repeat": (0.0, 0.5, 1.0), "copurchase": (0.0, 0.3, 0.6, 1.0), "popular": (0.2, 0.5, 1.0, 2.0)}
REASONS = ("you buy this often", "often bought with your items", "popular")


def _lines(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    done = orders[orders["is_completed"] == "true"][["order_id", "customer_id", "business_date"]]
    return items[["order_id", "product_id"]].drop_duplicates().merge(done, on="order_id")


def fit(lines: pd.DataFrame) -> dict:
    popularity = lines.groupby("product_id")["order_id"].nunique()
    personal = lines.groupby(["customer_id", "product_id"])["order_id"].nunique().rename("n").reset_index()
    pairs = lines[["order_id", "product_id"]].merge(lines[["order_id", "product_id"]], on="order_id")
    pairs = pairs[pairs["product_id_x"] != pairs["product_id_y"]]
    co = pairs.groupby(["product_id_x", "product_id_y"]).size().rename("co").reset_index()
    co["strength"] = co["co"] / popularity.reindex(co["product_id_y"]).to_numpy() ** 0.5   # damp popular partners
    co = co.sort_values("strength", ascending=False).groupby("product_id_x").head(PARTNERS_PER_PRODUCT)
    return {"popularity": popularity / popularity.max(), "personal": personal, "co": co}


def candidates(model: dict, customers: list[str]) -> pd.DataFrame:
    """One row per (customer, candidate product) with the three normalised signals."""
    pop = model["popularity"].sort_values(ascending=False)
    personal = model["personal"][model["personal"]["customer_id"].isin(customers)].copy()
    personal["repeat"] = personal["n"] / personal.groupby("customer_id")["n"].transform("max")
    co = (personal[["customer_id", "product_id"]].merge(model["co"], left_on="product_id", right_on="product_id_x")
          .groupby(["customer_id", "product_id_y"])["strength"].sum().rename("copurchase").reset_index()
          .rename(columns={"product_id_y": "product_id"}))
    co["copurchase"] /= co.groupby("customer_id")["copurchase"].transform("max")
    co = co.sort_values("copurchase", ascending=False).groupby("customer_id").head(N_COPURCHASE)
    popular = pd.DataFrame(list(product(customers, pop.index[:N_POPULAR])), columns=["customer_id", "product_id"])
    cand = (pd.concat([personal[["customer_id", "product_id"]], co[["customer_id", "product_id"]], popular])
            .drop_duplicates()
            .merge(personal[["customer_id", "product_id", "repeat"]], how="left", on=["customer_id", "product_id"])
            .merge(co, how="left", on=["customer_id", "product_id"]))
    cand[["repeat", "copurchase"]] = cand[["repeat", "copurchase"]].fillna(0.0)
    cand["popular"] = pop.reindex(cand["product_id"]).fillna(0.0).to_numpy()
    return cand


def rank(cand: pd.DataFrame, method: str = "hybrid", weights: dict | None = None) -> pd.DataFrame:
    if method == "popularity":
        score = cand["popular"]
    elif method == "repeat":
        score = cand["repeat"] + 0.01 * cand["popular"]
    else:
        w = weights or DEFAULT_WEIGHTS
        score = w["repeat"] * cand["repeat"] + w["copurchase"] * cand["copurchase"] + w["popular"] * cand["popular"]
    out = cand.assign(score=score.round(4)).sort_values(["customer_id", "score", "product_id"],
                                                         ascending=[True, False, True], kind="stable")
    out = out.groupby("customer_id").head(TOP_K).copy()
    out["rank"] = out.groupby("customer_id").cumcount() + 1
    out["reason"] = np.select([out["repeat"] > 0, out["copurchase"] > 0], REASONS[:2], REASONS[2])
    return out[["customer_id", "product_id", "rank", "score", "reason"]].reset_index(drop=True)


def recommend(model: dict, customers: list[str], method: str = "hybrid", weights: dict | None = None) -> pd.DataFrame:
    return rank(candidates(model, customers), method, weights)


def evaluate(recs: pd.DataFrame, actual: pd.DataFrame, catalog_size: int) -> dict:
    truth = actual.groupby("customer_id")["product_id"].apply(set)
    hits, precision, recall = [], [], []
    for c, grp in recs.groupby("customer_id"):
        bought = truth.get(c)
        if not bought:
            continue
        h = len(set(grp["product_id"]) & bought)
        hits.append(h > 0)
        precision.append(h / TOP_K)
        recall.append(h / len(bought))
    return {"precision_at_10": round(float(np.mean(precision)), 4), "recall_at_10": round(float(np.mean(recall)), 4),
            "hit_rate": round(float(np.mean(hits)), 4), "customers": len(precision),
            "coverage": round(recs["product_id"].nunique() / catalog_size, 4)}


def _split(lines: pd.DataFrame, end: pd.Timestamp, until: pd.Timestamp | None = None):
    train = lines[lines["business_date"] <= end]
    test = lines[(lines["business_date"] > end) & ((lines["business_date"] <= until) if until is not None else True)]
    return train, test, sorted(set(train["customer_id"]) & set(test["customer_id"]))


def choose_weights(lines: pd.DataFrame, train_end: pd.Timestamp, catalog_size: int) -> tuple[dict, pd.DataFrame]:
    """Score WEIGHT_GRID on the month before the test month (train on everything before it)."""
    val_end = train_end - pd.offsets.MonthEnd(1)
    train, val, customers = _split(lines, val_end, train_end)
    if not customers:   # short data (e.g. the 30-day test profile): no validation month -> default weights
        return dict(DEFAULT_WEIGHTS), pd.DataFrame(columns=[*WEIGHT_GRID, "precision_at_10", "recall_at_10"])
    cand = candidates(fit(train), customers)
    rows = []
    for wr, wc, wp in product(*WEIGHT_GRID.values()):
        w = {"repeat": wr, "copurchase": wc, "popular": wp}
        rows.append({**w, **evaluate(rank(cand, "hybrid", w), val, catalog_size)})
    grid = pd.DataFrame(rows).sort_values(["precision_at_10", "recall_at_10"], ascending=False, kind="stable")
    best = grid.iloc[0]
    return {k: float(best[k]) for k in WEIGHT_GRID}, grid


def backtest_and_fit(orders: pd.DataFrame, items: pd.DataFrame, train_end, catalog_size: int) -> dict:
    lines = _lines(orders, items)
    lines["business_date"] = pd.to_datetime(lines["business_date"])
    end = pd.Timestamp(train_end)
    weights, grid = choose_weights(lines, end, catalog_size)
    train, test, customers = _split(lines, end)
    cand = candidates(fit(train), customers)
    label = f"repeat {weights['repeat']:g} · co-purchase {weights['copurchase']:g} · popular {weights['popular']:g}"
    metrics = [{"method": "hybrid", "weights": label, **evaluate(rank(cand, "hybrid", weights), test, catalog_size)},
               {"method": "repeat", "weights": "buy again only", **evaluate(rank(cand, "repeat"), test, catalog_size)},
               {"method": "popularity", "weights": "top sellers", **evaluate(rank(cand, "popularity"), test, catalog_size)}]
    final = recommend(fit(lines), sorted(lines["customer_id"].unique()), "hybrid", weights)
    return {"metrics": pd.DataFrame(metrics), "recommendations": final, "weights": weights, "grid": grid}
