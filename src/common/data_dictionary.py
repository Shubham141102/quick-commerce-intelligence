"""Data dictionary: human documentation for every source column, rendered to docs/data_dictionary.md.

Schemas (types, nullability, keys) come from `schemas.py`; volumes and injected
issues come from the configs. Only descriptions, valid ranges and generation
notes are written here. `tests/unit/test_data_dictionary.py` fails if a column is
undocumented or the rendered doc is stale.

    python -m scripts.build_docs
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.config import Config, derived_targets
from src.common.schemas import (
    BRONZE_METADATA_COLUMNS,
    CORRUPT_RECORD,
    DIMENSION_DATASETS,
    SOURCE_SCHEMAS,
    STREAM_DATASETS,
)


@dataclass(frozen=True)
class ColumnDoc:
    description: str
    valid: str = ""
    fk: str = ""          # "table.column"


@dataclass(frozen=True)
class DatasetDoc:
    description: str
    grain: str
    event_time: str        # column that decides the landing file (business date in IST)
    generation: str


DATASETS: dict[str, DatasetDoc] = {
    "categories": DatasetDoc(
        "Product categories of the catalog.", "One row per category.", "— (dimension)",
        "Fixed list of quick-commerce categories; the profile decides how many are used."),
    "products": DatasetDoc(
        "Product catalog.", "One row per product.", "— (dimension)",
        "Spread evenly across categories. Popularity follows Zipf (s≈1.1) over a random ranking; the "
        "top `focus_skus_per_store` products are focus SKUs with tracked inventory. Fictional brands. "
        "Prices are constant during the period (price history is a Tier 2 option)."),
    "stores": DatasetDoc(
        "Dark stores.", "One row per store.", "— (dimension)",
        "`stores_per_city` stores per city, placed within `store_radius_km` of the city centre. Each "
        "store has a hidden demand factor (lognormal) that scales its order volume."),
    "customers": DatasetDoc(
        "Registered customers.", "One row per customer.", "`signup_date` (pre-period customers land in the first file)",
        "Each customer has a hidden persona (ground truth only) that drives order frequency, basket "
        "size, category mix, night ordering and promo sensitivity. 75% signed up before the period. "
        "Synthetic names (Faker, en_IN)."),
    "delivery_partners": DatasetDoc(
        "Delivery riders.", "One row per partner.", "— (dimension)",
        "`partners_per_store` partners per store; only `active` partners are assigned deliveries. "
        "Every store has at least one active partner."),
    "orders": DatasetDoc(
        "Customer orders.", "One row per order.", "`order_ts`",
        "The exact order count is allocated over store-days by store size × day-of-week × rainfall × "
        "trend (planted outages and spikes adjust it). Customers come from the store's city 97% of the "
        "time. Cancellation stage is decided up front: post-dispatch cancellations are twice as likely "
        "on days with more than 10 mm of rain."),
    "order_items": DatasetDoc(
        "Order lines.", "One row per product in an order.", "parent order's `order_ts`",
        "Basket size per persona, nudged so the total hits the target. Category by persona mix (boosted by "
        "active promotions), product by Zipf within the category; products are distinct within an "
        "order. Planted affinity pairs and demand spikes are applied, then out-of-stock focus SKUs are "
        "substituted or reduced by the inventory simulation."),
    "payments": DatasetDoc(
        "Payment attempts.", "One row per attempt.", "`payment_ts`",
        "One final attempt per order; ~5% of prepaid orders have an earlier failed attempt, concentrated "
        "in planted payment-gateway failure windows. COD is settled at delivery time. Pre-dispatch "
        "cancellations of prepaid orders are `voided`."),
    "deliveries": DatasetDoc(
        "Deliveries.", "One row per dispatched order.", "`pickup_ts`",
        "Every order except pre-dispatch cancellations. Preparation ~4 min (lognormal); travel "
        "lognormal with the configured median, slower in rain. Post-dispatch cancellations end "
        "`cancelled` or `failed` with no `delivered_ts`."),
    "cancellations": DatasetDoc(
        "Order cancellations.", "One row per cancelled order.", "`cancelled_ts`",
        "Pre-dispatch: 1–6 min after the order. Post-dispatch: 3–20 min after pickup."),
    "inventory_snapshots": DatasetDoc(
        "Weekly stock counts for focus SKUs.", "One row per store × focus SKU × snapshot time.", "`snapshot_ts`",
        "Taken every snapshot weekday at the configured hour (IST) from the inventory simulation, so "
        "the counts match the simulated stock exactly."),
    "inventory_events": DatasetDoc(
        "Stock movements for focus SKUs.", "One row per event.", "`event_ts`",
        "Restocks (nightly check at 22:00 IST when stock ≤ reorder level, arriving 1–2 days later at 07:00), "
        "random damage and cycle-count adjustments, write-offs from planted stockout episodes, and a few "
        "end-of-period adjustments that make the count hit the target. Sales are not events; they come from orders."),
    "application_logs": DatasetDoc(
        "Service logs.", "One row per log line.", "`event_ts`",
        "Background logs across five services, plus bursts of payment-service ERRORs during planted "
        "gateway failures. Messages contain commas and quotes on purpose (CSV quoting test)."),
    "promotions": DatasetDoc(
        "Category promotions.", "One row per promotion.", "— (dimension)",
        "Each promotion discounts one category (popular categories are more likely) for 3–14 days. "
        "Active promotions raise their category's share of baskets."),
    "order_promotions": DatasetDoc(
        "Promotions applied to orders.", "One row per order × promotion (at most one promotion per order).",
        "parent order's `order_ts`",
        "Chosen among orders that contain a category with an active promotion, weighted by persona promo "
        "sensitivity; the best eligible discount applies to that category's lines."),
    "returns_refunds": DatasetDoc(
        "Refunds.", "One row per refund.", "`refund_ts`",
        "Full refunds for post-dispatch cancellations of prepaid orders, then partial refunds (one line) for "
        "delivered orders: damaged, missing, poor quality or expired items."),
    "reviews": DatasetDoc(
        "Product reviews.", "One row per review.", "`review_ts`",
        "For delivered orders; one product from the order. Ratings skew positive, and lower when delivery "
        "took more than SLA + 10 min. 30% have no comment (valid)."),
    "application_events": DatasetDoc(
        "App clickstream.", "One row per event.", "`event_ts`",
        "Sessions of 3–7 steps leading up to real orders (logged-in), plus anonymous browsing sessions "
        "of 2–4 steps with no customer_id."),
    "weather": DatasetDoc(
        "City weather observations.", "One row per city × observation time.", "`observation_ts`",
        "Readings every 3 hours. Rainy-day probability by month (monsoon June–September), gamma-distributed "
        "daily rainfall split across readings; temperature has a daily cycle and cools with rain."),
}

STATUS = "Canonical lowercase"
COLUMNS: dict[str, dict[str, ColumnDoc]] = {
    "categories": {
        "category_id": ColumnDoc("Category identifier.", "`CAT01`…"),
        "category_name": ColumnDoc("Category display name.", "Non-empty"),
    },
    "products": {
        "product_id": ColumnDoc("Product identifier.", "`P0001`…"),
        "product_name": ColumnDoc("Brand + item + pack size.", "Non-empty"),
        "category_id": ColumnDoc("Product category.", "", "categories.category_id"),
        "brand": ColumnDoc("Fictional brand name.", "Non-empty"),
        "price": ColumnDoc("Selling price in INR.", "> 0; generated 10–1,200"),
    },
    "stores": {
        "store_id": ColumnDoc("Store identifier.", "`S01`…"),
        "store_name": ColumnDoc("Store display name.", "Non-empty"),
        "city_id": ColumnDoc("City identifier (configs/cities.yaml).", "`C01`–`C03`"),
        "city": ColumnDoc("City name.", "Matches city_id"),
        "state": ColumnDoc("State name.", "Matches city_id"),
        "latitude": ColumnDoc("Store latitude.", "−90…90; within store_radius_km of the city centre"),
        "longitude": ColumnDoc("Store longitude.", "−180…180"),
    },
    "customers": {
        "customer_id": ColumnDoc("Customer identifier.", "`CUST00001`…"),
        "name": ColumnDoc("Synthetic full name.", "Non-empty"),
        "email": ColumnDoc("Email address; optional (3% missing is valid).", "Valid email format when present"),
        "city_id": ColumnDoc("Home city.", "`C01`–`C03`"),
        "signup_date": ColumnDoc("Registration date.", "2023-01-01 … period end"),
    },
    "delivery_partners": {
        "partner_id": ColumnDoc("Partner identifier.", "`DP0001`…"),
        "partner_name": ColumnDoc("Synthetic full name.", "Non-empty"),
        "store_id": ColumnDoc("Home store.", "", "stores.store_id"),
        "city_id": ColumnDoc("City of the home store.", "Matches the store's city"),
        "availability_status": ColumnDoc("Current availability.", f"{STATUS}: `active`, `on_leave`, `inactive`"),
    },
    "orders": {
        "order_id": ColumnDoc("Order identifier.", "`ORD0000001`…"),
        "customer_id": ColumnDoc("Ordering customer.", "", "customers.customer_id"),
        "store_id": ColumnDoc("Fulfilling store.", "", "stores.store_id"),
        "order_ts": ColumnDoc("When the order was placed (UTC).", "Within the period; ≥ customer signup_date"),
        "status": ColumnDoc("Final order status.", f"{STATUS}: `delivered`, `cancelled`"),
        "total_amount": ColumnDoc("Amount payable in INR.", "≥ 0; = Σ(quantity × unit_price) − order discount"),
    },
    "order_items": {
        "order_item_id": ColumnDoc("Line identifier.", "`OI00000001`…"),
        "order_id": ColumnDoc("Parent order.", "", "orders.order_id"),
        "product_id": ColumnDoc("Product ordered; distinct within an order.", "", "products.product_id"),
        "quantity": ColumnDoc("Units ordered.", "Integer ≥ 1; usually 1–5, outlier orders 8–20"),
        "unit_price": ColumnDoc("Price per unit in INR.", "= products.price"),
    },
    "payments": {
        "payment_id": ColumnDoc("Payment attempt identifier.", "`PAY0000001`…"),
        "order_id": ColumnDoc("Order being paid.", "", "orders.order_id"),
        "attempt_no": ColumnDoc("Attempt sequence within the order.", "1 or 2"),
        "method": ColumnDoc("Payment method.", f"{STATUS}: `upi`, `card`, `wallet`, `cod`"),
        "amount": ColumnDoc("Amount in INR.", "= orders.total_amount"),
        "status": ColumnDoc("Attempt outcome.", f"{STATUS}: `success`, `failed`, `voided`, `cancelled`; at most one `success` per order"),
        "payment_ts": ColumnDoc("When the attempt happened (UTC).", "Failed attempts precede the final attempt; COD = delivery time"),
    },
    "deliveries": {
        "delivery_id": ColumnDoc("Delivery identifier.", "`DEL0000001`…"),
        "order_id": ColumnDoc("Delivered order; one delivery per order.", "", "orders.order_id"),
        "partner_id": ColumnDoc("Assigned rider.", "Rider from the same city as the store", "delivery_partners.partner_id"),
        "pickup_ts": ColumnDoc("When the rider picked up the order (UTC).", "> orders.order_ts"),
        "delivered_ts": ColumnDoc("When the order was handed over (UTC); empty unless delivered.", "> pickup_ts"),
        "status": ColumnDoc("Delivery outcome.", f"{STATUS}: `delivered`, `cancelled`, `failed`"),
    },
    "cancellations": {
        "cancellation_id": ColumnDoc("Cancellation identifier.", "`CAN000001`…"),
        "order_id": ColumnDoc("Cancelled order.", "", "orders.order_id"),
        "stage": ColumnDoc("When it was cancelled.", "`pre_dispatch` (no delivery row) or `post_dispatch`"),
        "reason": ColumnDoc("Cancellation reason code.", f"{STATUS}: e.g. `customer_changed_mind`, `delivery_delayed`"),
        "cancelled_ts": ColumnDoc("When it was cancelled (UTC).", "> orders.order_ts"),
    },
    "inventory_snapshots": {
        "snapshot_id": ColumnDoc("Snapshot row identifier.", "`SNP000001`…"),
        "store_id": ColumnDoc("Store.", "", "stores.store_id"),
        "product_id": ColumnDoc("Focus SKU.", "", "products.product_id"),
        "stock_quantity": ColumnDoc("Units on hand at snapshot time.", "Integer ≥ 0"),
        "reorder_level": ColumnDoc("Restock trigger level for the pair.", "Integer ≥ 1"),
        "snapshot_ts": ColumnDoc("Count time (UTC).", "Snapshot weekday at the configured IST hour"),
    },
    "inventory_events": {
        "event_id": ColumnDoc("Event identifier.", "`IEV000001`…"),
        "store_id": ColumnDoc("Store.", "", "stores.store_id"),
        "product_id": ColumnDoc("Focus SKU.", "", "products.product_id"),
        "event_type": ColumnDoc("Kind of stock movement.", f"{STATUS}: `restock`, `damage`, `adjustment`"),
        "quantity": ColumnDoc("Signed change in units.", "restock > 0, damage < 0, adjustment ≠ 0"),
        "event_ts": ColumnDoc("When stock changed (UTC).", "Within the period"),
    },
    "application_logs": {
        "log_id": ColumnDoc("Log line identifier.", "`LOG000001`…"),
        "event_ts": ColumnDoc("Log time (UTC).", "Within the period"),
        "service": ColumnDoc("Emitting service.", "`order-service`, `payment-service`, `inventory-service`, `delivery-service`, `search-service`"),
        "log_level": ColumnDoc("Severity.", "Canonical uppercase: `INFO`, `WARN`, `ERROR`, `DEBUG`"),
        "message": ColumnDoc("Free-text message; may contain commas and quotes.", "Non-empty"),
    },
    "promotions": {
        "promotion_id": ColumnDoc("Promotion identifier.", "`PR001`…"),
        "name": ColumnDoc("Display name.", "Non-empty"),
        "category_id": ColumnDoc("Discounted category.", "", "categories.category_id"),
        "discount_pct": ColumnDoc("Discount percentage.", "0 < x ≤ 100; generated 5–30"),
        "start_date": ColumnDoc("First active day (IST).", "≤ end_date"),
        "end_date": ColumnDoc("Last active day (IST).", "≥ start_date"),
    },
    "order_promotions": {
        "order_id": ColumnDoc("Discounted order.", "", "orders.order_id"),
        "promotion_id": ColumnDoc("Applied promotion; active on the order's IST date.", "", "promotions.promotion_id"),
        "discount_amount": ColumnDoc("Discount in INR.", "> 0; = discount_pct × the category's line amounts"),
    },
    "returns_refunds": {
        "refund_id": ColumnDoc("Refund identifier.", "`REF000001`…"),
        "payment_id": ColumnDoc("Refunded (successful) payment.", "", "payments.payment_id"),
        "order_id": ColumnDoc("Order of the payment.", "", "orders.order_id"),
        "amount": ColumnDoc("Refund in INR.", "> 0; ≤ the payment amount"),
        "reason": ColumnDoc("Refund reason code.", f"{STATUS}: `order_cancelled`, `damaged_item`, `missing_item`, …"),
        "status": ColumnDoc("Refund state.", f"{STATUS}: `completed`, `pending`, `rejected`"),
        "refund_ts": ColumnDoc("When the refund was raised (UTC).", "> payments.payment_ts"),
    },
    "reviews": {
        "review_id": ColumnDoc("Review identifier.", "`REV000001`…"),
        "customer_id": ColumnDoc("Reviewer.", "", "customers.customer_id"),
        "product_id": ColumnDoc("Reviewed product; bought in `order_id`.", "", "products.product_id"),
        "order_id": ColumnDoc("Order the product came from.", "", "orders.order_id"),
        "rating": ColumnDoc("Star rating.", "Integer 1–5"),
        "comment": ColumnDoc("Free text; optional (30% missing is valid and is not a negative signal).", ""),
        "review_ts": ColumnDoc("When the review was written (UTC).", "> deliveries.delivered_ts"),
    },
    "application_events": {
        "event_id": ColumnDoc("Event identifier.", "`EVT0000001`…"),
        "customer_id": ColumnDoc("Logged-in customer; empty for anonymous sessions (~15%, valid).", "", "customers.customer_id"),
        "session_id": ColumnDoc("App session.", "`SES0000001`…"),
        "event_name": ColumnDoc("Screen / action.", f"{STATUS}: `app_open`, `search`, `view_product`, `add_to_cart`, `view_cart`, `checkout_start`, `payment_page`"),
        "event_ts": ColumnDoc("Event time (UTC).", "Within the period"),
        "metadata": ColumnDoc("JSON object as a string.", "Valid JSON; keys `platform`, `app_version`, `screen`, optional `query`, `product_id`"),
    },
    "weather": {
        "city_id": ColumnDoc("City (join to stores via city_id).", "`C01`–`C03`"),
        "observation_ts": ColumnDoc("Reading time (UTC).", "Every 3 hours (IST-aligned)"),
        "temperature_c": ColumnDoc("Air temperature in °C.", "Plausible range 0–50"),
        "rainfall_mm": ColumnDoc("Rain since the previous reading, mm.", "≥ 0"),
        "humidity_pct": ColumnDoc("Relative humidity, %.", "0–100"),
    },
}

BRONZE_COLUMNS_DOC = {
    "_corrupt_record": "Raw line when Spark could not parse the row (wrong number of fields); empty otherwise.",
    "_batch_id": "Bronze batch: `<run_id>__historical`, `<run_id>__batch` or `<run_id>__stream_<epoch>`.",
    "_pipeline_run_id": "Pipeline run that loaded the row (joins to `meta_pipeline_runs`).",
    "_source_file": "Landing file the row came from: `<generation_run>/landing/<slice>/<dataset>/<file>.csv`.",
    "_load_type": "`historical`, `batch` or `stream`.",
    "_source_system": "Simulated source system (`oms`, `wms`, `crm`, `catalog`, `payments`, `logistics`, `app`, ...).",
    "_schema_version": "Source schema version.",
    "_ingestion_ts": "When the batch was written to Bronze (UTC).",
    "_ingestion_date": "Ingestion date in IST.",
    "_record_hash": "SHA-256 of the source columns + `_corrupt_record`; identical raw rows have identical hashes.",
    "_is_late": "`true` if the event time is before the window its landing file covers (earlier day / hour); "
                "empty when not applicable or the timestamp can't be parsed.",
    "_beyond_watermark": "Stream rows only: `true` if older than the 2-hour watermark (max event time of earlier "
                         "micro-batches − 2 h), i.e. a watermarked streaming aggregation would drop it.",
}

ISSUE_NAMES = {
    "DUP": "Duplicate", "MISS": "Missing required value", "FMT": "Inconsistent format",
    "NUM": "Invalid numeric / out of range", "BIZ": "Business-rule violation", "TYPE": "Invalid data type",
    "FK": "Referential-integrity error", "SEQ": "Invalid temporal sequence", "MALF": "Malformed payload",
}


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(cfg: Config) -> str:
    targets = derived_targets(cfg)
    dirty = cfg.dirty
    out: list[str] = []
    w = out.append

    w("# Data Dictionary — Source Datasets")
    w("")
    w("> Generated by `python -m scripts.build_docs` from `src/common/schemas.py`, "
      "`src/common/data_dictionary.py` and `configs/`. Do not edit by hand.")
    w("")
    w(f"Volumes are the **{cfg.profile}** profile generation targets (before duplicate copies are added). "
      f"Business period: {cfg.calendar.start_date} → {cfg.calendar.end_date}.")
    w("")
    w("## Conventions")
    w("")
    w("- **Storage:** every table is CSV (UTF-8, comma, header, RFC 4180 quoting). Bronze reads every column as string; "
      "Silver casts to the logical type below.")
    w("- **Types:** `string`, `int`, `decimal` (money, 2 dp, read as decimal(12,2)), `double`, "
      "`timestamp` (ISO-8601 UTC, `yyyy-MM-ddTHH:mm:ssZ`), `date` (`yyyy-MM-dd`).")
    w("- **Nulls:** written as an empty field; empty string and null are the same thing.")
    w("- **Time:** timestamps are stored in UTC; the business date is the date in `Asia/Kolkata` (IST). "
      "Landing files are split by the IST date of each dataset's event-time column.")
    w("- **Required:** `yes` means a missing value is a data-quality failure (MISS); `no` means missing is valid.")
    w("- **Names:** Bronze table `brz_<dataset>`, Silver table `slv_<dataset>`.")
    w("- **Dirty data:** every dataset is generated 90% clean; the issue types injected into each dataset are listed "
      "per table (see Project_Plan_v2.md §4.2).")
    w("")
    w("## Overview")
    w("")
    w("| Dataset | Tier | Grain | Business key | Rows | Landing | Bronze | Silver |")
    w("|---|:-:|---|---|---:|---|---|---|")
    for name, schema in SOURCE_SCHEMAS.items():
        doc = DATASETS[name]
        landing = ("historical (one file)" if name in DIMENSION_DATASETS
                   else "historical / batch / stream" if name in STREAM_DATASETS else "historical / batch")
        w(f"| [{name}](#{name.replace('_', '_')}) | {schema.tier} | {_md_escape(doc.grain)} | "
          f"`{', '.join(schema.business_key)}` | {targets[name]:,} | {landing} | `brz_{name}` | `slv_{name}` |")
    w("")
    w("## Relationships")
    w("")
    w("| From | To |")
    w("|---|---|")
    for name in SOURCE_SCHEMAS:
        for col, cd in COLUMNS[name].items():
            if cd.fk:
                w(f"| `{name}.{col}` | `{cd.fk}` |")
    w("| `stores.city_id`, `customers.city_id`, `delivery_partners.city_id`, `weather.city_id` | `configs/cities.yaml` city_id |")
    w("")
    w("## Bronze columns added to every table")
    w("")
    w("Bronze table `brz_<dataset>` = the source columns exactly as received (all strings) + these columns. "
      "Stored under `data/bronze/brz_<dataset>/batch=<batch_id>/`.")
    w("")
    w("| Column | Description |")
    w("|---|---|")
    for col in [CORRUPT_RECORD, *BRONZE_METADATA_COLUMNS]:
        w(f"| `{col}` | {_md_escape(BRONZE_COLUMNS_DOC[col])} |")
    w("")

    for name, schema in SOURCE_SCHEMAS.items():
        doc = DATASETS[name]
        w(f"## {name}")
        w("")
        w(f"{doc.description} **Grain:** {doc.grain} **Business key:** `{', '.join(schema.business_key)}`. "
          f"**Tier:** {schema.tier}. **Rows ({cfg.profile}):** {targets[name]:,}. "
          f"**Landing file decided by:** {doc.event_time}.")
        w("")
        w(f"*Generation:* {doc.generation}")
        w("")
        w("| Column | Type | Required | Description | Valid values / rule | References |")
        w("|---|---|:-:|---|---|---|")
        for col in schema.columns:
            cd = COLUMNS[name][col.name]
            key = " 🔑" if col.name in schema.business_key else ""
            w(f"| `{col.name}`{key} | {col.dtype} | {'no' if col.nullable else 'yes'} | "
              f"{_md_escape(cd.description)} | {_md_escape(cd.valid)} | {f'`{cd.fk}`' if cd.fk else ''} |")
        w("")
        mutations = dirty["mutations"].get(name, {})
        issues = []
        for code in dirty["applicability"][name]:
            detail = ", ".join(f"`{m}`" for m in mutations.get(code, []))
            if code == "DUP":
                detail = "exact copies and formatting-only variants of untouched rows"
            issues.append(f"{ISSUE_NAMES[code]} (`{code}`){': ' + detail if detail else ''}")
        w("*Injected issues:* " + "; ".join(issues) + ".")
        w("")
    return "\n".join(out)
