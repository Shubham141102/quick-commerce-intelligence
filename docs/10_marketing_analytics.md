# 10. Marketing Analytics — Model Cards (Phase 5D)

**Status:** complete · **Code:** `src/ml/marketing.py` (orchestration, basket rules, retention), `src/ml/segmentation.py`, `src/ml/recommendations.py` · **Run:** `python -m scripts.run_pipeline --stages ml,publish` · **Check:** `python -m scripts.check_phase5` (section 5D)
**Outputs (columns in [gold_catalog.md](gold_catalog.md)):** `gld_customer_segments`, `gld_segment_profiles`, `gld_segmentation_selection`, `gld_basket_rules`, `gld_recommendations`, `gld_recommendation_metrics`, `gld_retention_cohorts`, `gld_customer_retention`
**Registry:** `meta_model_runs` (`customer_segmentation`, `recommendations`), `meta_model_metrics` (backtest and the August weight grid)

The step runs in pandas / scikit-learn inside the `ml` stage (no Spark). Ground truth (`gt_personas`, `gt_affinity_pairs`) is read **only** by the check script.

## Pass bars (fixed before measuring, 2026-10-05)

| Check | Bar | Result |
|---|---|---|
| 5D.1 Every buyer in one segment; k in 3–8; unique names | structural | ✅ PASS: k = 7, 5,307 customers |
| 5D.2 Segments recover the planted personas | ARI ≥ 0.3 | ✅ PASS: **0.505** (5 personas vs 7 segments) |
| 5D.3 Stability across random seeds | ARI ≥ 0.7 | ✅ PASS: **0.840** |
| 5D.4 Planted affinity pairs found as rules A → B | ≥ 20 of 25 | ✅ PASS: **23 of 25** among 352 rules |
| 5D.5 Recommendations beat popularity (Precision@10, Sept) | hybrid > popularity | ✅ PASS after one approved revision: **14.8% vs 14.0%** |
| 5D.6 Retention arithmetic | consistency | ✅ PASS |
| 5D.7 Marketing functions role-checked | 15/15 | ✅ PASS |

---

## Model card 1: Customer segmentation

| | |
|---|---|
| **Purpose** | Group customers by how they shop so campaigns can be targeted |
| **Population** | Customers with ≥ 1 completed order (5,307 of 6,000) |
| **Features** | log spend, log orders, recency, AOV, items per order, distinct categories, night / weekend / promotion share, cancellation rate, share of spend per category (from `gld_customer_360`, `gld_customer_category`), standardised |
| **Method** | K-means; k chosen from 3–8 by silhouette (best k = 7, silhouette 0.078); stability = mean adjusted Rand index vs 5 reruns with other seeds (0.84) |
| **Names** | Generated from each segment's measured traits (z-score > 0.5 on night / promotion / weekend / basket / frequency / spend; "Lapsing" if recency is high) plus its top category. Never preset names. A campaign idea is suggested from the same traits. |

**Segments found:**

| Segment | Customers | Avg spend | Avg orders | AOV | Night share | Campaign idea |
|---|---|---|---|---|---|---|
| Frequent · High-value — Atta, Rice & Dal | 1,385 (26%) | ₹7,059 | 9.8 | ₹722 | 18% | Loyalty perks; protect this revenue |
| Big-basket · Premium-basket — Baby Care | 1,034 (19%) | ₹7,921 | 4.9 | ₹1,680 | 18% | Loyalty perks and premium assortment |
| Everyday — Atta, Rice & Dal | 936 (18%) | ₹1,056 | 2.0 | ₹563 | 23% | Cross-sell to raise basket size |
| Night-time — Sweets & Chocolates | 878 (17%) | ₹2,418 | 6.1 | ₹410 | 64% | Late-evening push and snack bundles |
| Everyday — Gourmet & Organic | 699 (13%) | ₹4,711 | 4.7 | ₹1,094 | 26% | Cross-sell from top category |
| Everyday — Health & Wellness | 256 (5%) | ₹2,565 | 3.5 | ₹775 | 31% | Cross-sell from top category |
| Night-time — Instant & Frozen Food | 119 (2%) | ₹360 | 1.9 | ₹182 | 55% | Late-evening push and snack bundles |

**Limitations.** The silhouette is low (0.08): shoppers form overlapping groups rather than sharply separated ones, which is normal for behavioural data. The segments are still stable and agree with the planted personas (ARI 0.5). Campaign ideas are suggestions, not tested results. One customer nets to −₹34.80 spend in `gld_customer_360` (refund larger than order value) and is treated as ₹0 spend; worth a Gold follow-up.

## Model card 2: Basket rules (affinity)

| | |
|---|---|
| **Purpose** | "When A is in the basket, B often is too": placement, bundles, checkout suggestions |
| **Input** | `gld_basket_pairs` (completed orders) |
| **Method** | Each pair is turned into two directed rules A → B and B → A, with confidence = baskets with both ÷ baskets with A, and lift = confidence ÷ share of baskets with B. Rules are kept when confidence ≥ 10% and lift ≥ 2 (thresholds set in advance). |
| **Result** | 352 rules; 23 of the 25 planted affinity pairs found in the right direction |
| **Limitations** | Pairs only (no 3-item sets); rare products can show high lift on few baskets, so `baskets_both` is shown next to every rule |

## Model card 3: Recommendations

| | |
|---|---|
| **Purpose** | Top-10 products per customer, each with a reason ("you buy this often" / "often bought with your items" / "popular") |
| **Input** | Silver completed orders and items |
| **Score** | w_repeat × own purchase frequency + w_copurchase × co-purchase strength with the customer's items + w_popular × global popularity (each normalised 0–1) |
| **Weights** | Chosen on **August** from a grid fixed in advance (repeat {0, 0.5, 1} × co-purchase {0, 0.3, 0.6, 1} × popular {0.2, 0.5, 1, 2}), training on Apr–Jul. Chosen: **repeat 0.5 · co-purchase 1 · popular 1** |
| **Test** | Refit on Apr–Aug and score **September once**; 2,670 customers who bought in both periods |

| Method (September) | Precision@10 | Recall@10 | Hit rate | Coverage |
|---|---|---|---|---|
| **Hybrid (used)** | **14.8%** | 28.0% | 73% | 34% |
| Popularity baseline | 14.0% | 26.5% | 72% | 2% |
| Buy-again only | 12.7% | 24.0% | 69% | 91% |

**Revision (approved 2026-10-05, after the first measurement).** The first version used fixed weights (repeat 1.0, co-purchase 0.6, popular 0.2) on the assumption that quick-commerce shoppers mostly re-buy the same items. It lost to popularity (13.4% vs 14.0%). A diagnostic showed only **27%** of September purchases were repeats, while the 10 best-sellers alone covered another 27.6%. The weights are now chosen on the validation month; September was never used to choose anything.

**Limitations.** The win over popularity is small (+0.8 points), but the hybrid recommends **17× more distinct products** (coverage 34% vs 2%), which matters for discovery. The final model is refit on all data for the app.

## Model card 4: Retention (descriptive only)

| | |
|---|---|
| **Purpose** | Who is still ordering and who has gone quiet; win-back lists |
| **Cohorts** | Customers grouped by the month of their first completed order; share with a completed order in each later month |
| **Status** | Days since last completed order at the end of the data: active ≤ 14, cooling 15–30, at risk 31–60, lapsed > 60, plus never ordered. Value tier = spend thirds; overdue = days since last order > 2 × the customer's own average gap |
| **No churn prediction** | The synthetic data has no planted churn signal, so a predictive model would have nothing true to learn. It is a Tier 2 item (Project_Plan_v2.md §10.1). |

**Results:** active 1,933 · cooling 1,143 · at risk 1,124 · lapsed 1,107 · never ordered 693; 1,050 customers overdue. Month-1 retention falls from 64% (April cohort) to 44% (July/August cohorts). After the first month the April cohort stays near 60%.
