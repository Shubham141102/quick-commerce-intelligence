"""Customer segmentation (Phase 5D): K-means on behaviour + category mix, segments named from their profile.

Customers with at least one completed order are clustered on standardised features (spend, frequency,
recency, basket size, night / weekend / promotion shares, cancellation rate, share of spend per category).
k is chosen from 3–8 by silhouette score; stability is the adjusted Rand index between runs with different
seeds. Labels are generated from each segment's most distinctive measured traits — never preset names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

K_RANGE = range(3, 9)
BEHAVIOUR = ["log_spend", "log_orders", "recency_days", "avg_order_value", "avg_items_per_order", "distinct_categories",
             "night_order_share", "weekend_order_share", "promo_order_share", "cancellation_rate"]
TRAIT_WORDS = {  # feature -> phrase used when the segment is clearly above average on it
    "night_order_share": "Night-time", "promo_order_share": "Deal-seeking", "weekend_order_share": "Weekend",
    "avg_items_per_order": "Big-basket", "log_orders": "Frequent", "log_spend": "High-value",
    "avg_order_value": "Premium-basket",
}


def build_features(c360: pd.DataFrame, cat: pd.DataFrame, category_names: dict[str, str]) -> pd.DataFrame:
    buyers = c360[c360["orders_completed"] > 0].copy()
    buyers["log_spend"] = np.log1p(buyers["total_spend"].astype(float).clip(lower=0))   # 1 customer nets below 0
    buyers["log_orders"] = np.log1p(buyers["orders_completed"].astype(float))
    buyers["recency_days"] = buyers["recency_days"].fillna(buyers["recency_days"].max())
    for col in BEHAVIOUR:
        buyers[col] = buyers[col].astype(float).fillna(0.0)
    shares = (cat.pivot_table(index="customer_id", columns="category_id", values="share_of_revenue", fill_value=0.0)
              .rename(columns=lambda c: f"cat:{c}"))
    feats = buyers.set_index("customer_id")[BEHAVIOUR].join(shares, how="left").fillna(0.0)
    feats.attrs["category_names"] = category_names
    return feats


def choose_k(X: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=seed)
        rows.append({"k": k, "silhouette": round(float(sil), 4), "inertia": round(float(km.inertia_), 1)})
    sel = pd.DataFrame(rows)
    sel["chosen"] = sel["silhouette"] == sel["silhouette"].max()
    return sel


def _label(z: pd.Series, top_category: str) -> str:
    traits = [TRAIT_WORDS[f] for f in z.sort_values(ascending=False).index
              if f in TRAIT_WORDS and z[f] > 0.5][:2]
    if z.get("recency_days", 0) > 0.8:
        traits.insert(0, "Lapsing")
    return f"{' · '.join(traits[:2]) if traits else 'Everyday'} — {top_category}"


def _campaign(z: pd.Series) -> str:
    if z.get("recency_days", 0) > 0.8:
        return "Win-back offer: they have not ordered recently."
    if z.get("promo_order_share", 0) > 0.5:
        return "Targeted category discounts; they respond to promotions."
    if z.get("night_order_share", 0) > 0.5:
        return "Late-evening push notifications and snack bundles."
    if z.get("log_spend", 0) > 0.5:
        return "Loyalty perks and premium assortment; protect this revenue."
    if z.get("avg_items_per_order", 0) > 0.5:
        return "Bundle and bulk-pack offers for weekly stock-ups."
    return "Cross-sell from their top category to raise basket size."


def segment(feats: pd.DataFrame, seed: int) -> dict:
    X = StandardScaler().fit_transform(feats.to_numpy())
    selection = choose_k(X, seed)
    k = int(selection.loc[selection["chosen"], "k"].iloc[0])
    model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = model.labels_
    stability = float(np.mean([adjusted_rand_score(labels, KMeans(n_clusters=k, n_init=20, random_state=seed + s)
                                                   .fit(X).labels_) for s in range(1, 6)]))
    selection["stability_ari"] = np.where(selection["chosen"], round(stability, 4), np.nan)

    names = feats.attrs["category_names"]
    cat_cols = [c for c in feats.columns if c.startswith("cat:")]
    overall_mean, overall_std = feats[BEHAVIOUR].mean(), feats[BEHAVIOUR].std().replace(0, 1)
    profiles, seg_label = [], {}
    for s in range(k):
        members = feats[labels == s]
        z = (members[BEHAVIOUR].mean() - overall_mean) / overall_std
        top = members[cat_cols].mean().sort_values(ascending=False).head(3)
        top_names = [names.get(c.split(":", 1)[1], c) for c in top.index]
        seg_label[s] = _label(z, top_names[0])
        profiles.append({
            "segment_id": s, "segment_label": seg_label[s], "customers": len(members),
            "share": round(len(members) / len(feats), 4),
            "avg_spend": round(float(np.expm1(members["log_spend"]).mean()), 2),
            "avg_orders": round(float(np.expm1(members["log_orders"]).mean()), 2),
            "avg_order_value": round(float(members["avg_order_value"].mean()), 2),
            "avg_items_per_order": round(float(members["avg_items_per_order"].mean()), 2),
            "avg_recency_days": round(float(members["recency_days"].mean()), 1),
            "night_order_share": round(float(members["night_order_share"].mean()), 4),
            "weekend_order_share": round(float(members["weekend_order_share"].mean()), 4),
            "promo_order_share": round(float(members["promo_order_share"].mean()), 4),
            "top_categories": " | ".join(f"{n} ({v:.0%})" for n, v in zip(top_names, top.values)),
            "campaign_idea": _campaign(z),
        })
    profiles = pd.DataFrame(profiles)
    # make labels unique (two segments can share the same traits)
    dup = profiles["segment_label"].duplicated(keep=False)
    profiles.loc[dup, "segment_label"] = profiles.loc[dup].apply(
        lambda r: f"{r['segment_label']} ({r['segment_id'] + 1})", axis=1)
    seg_label = dict(zip(profiles["segment_id"], profiles["segment_label"]))
    assignments = pd.DataFrame({"customer_id": feats.index, "segment_id": labels})
    assignments["segment_label"] = assignments["segment_id"].map(seg_label)
    return {"assignments": assignments, "profiles": profiles, "selection": selection, "k": k,
            "silhouette": float(selection.loc[selection["chosen"], "silhouette"].iloc[0]), "stability": stability}
