"""Source dataset schemas: column order, logical types, nullability and business keys.

These are the single definition of each source table. The generator writes
columns in this order; Bronze reads every column as string; Silver casts to
`dtype`. Logical types: string, int, decimal, double, timestamp, date.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    nullable: bool = False


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[Column, ...]
    business_key: tuple[str, ...]
    tier: int = 1

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name} has no column {name!r}")


def _t(name: str, key: tuple[str, ...], *cols: tuple, tier: int = 1) -> TableSchema:
    return TableSchema(name, tuple(Column(*c) for c in cols), key, tier)


SOURCE_SCHEMAS: dict[str, TableSchema] = {
    s.name: s
    for s in [
        _t("categories", ("category_id",),
           ("category_id", "string"), ("category_name", "string")),
        _t("products", ("product_id",),
           ("product_id", "string"), ("product_name", "string"), ("category_id", "string"),
           ("brand", "string"), ("price", "decimal")),
        _t("stores", ("store_id",),
           ("store_id", "string"), ("store_name", "string"), ("city_id", "string"),
           ("city", "string"), ("state", "string"), ("latitude", "double"), ("longitude", "double")),
        _t("customers", ("customer_id",),
           ("customer_id", "string"), ("name", "string"), ("email", "string", True),
           ("city_id", "string"), ("signup_date", "date")),
        _t("delivery_partners", ("partner_id",),
           ("partner_id", "string"), ("partner_name", "string"), ("store_id", "string"),
           ("city_id", "string"), ("availability_status", "string")),
        _t("orders", ("order_id",),
           ("order_id", "string"), ("customer_id", "string"), ("store_id", "string"),
           ("order_ts", "timestamp"), ("status", "string"), ("total_amount", "decimal")),
        _t("order_items", ("order_id", "product_id"),
           ("order_item_id", "string"), ("order_id", "string"), ("product_id", "string"),
           ("quantity", "int"), ("unit_price", "decimal")),
        _t("payments", ("payment_id",),
           ("payment_id", "string"), ("order_id", "string"), ("attempt_no", "int"),
           ("method", "string"), ("amount", "decimal"), ("status", "string"),
           ("payment_ts", "timestamp")),
        _t("deliveries", ("delivery_id",),
           ("delivery_id", "string"), ("order_id", "string"), ("partner_id", "string"),
           ("pickup_ts", "timestamp"), ("delivered_ts", "timestamp", True), ("status", "string")),
        _t("cancellations", ("cancellation_id",),
           ("cancellation_id", "string"), ("order_id", "string"), ("stage", "string"),
           ("reason", "string"), ("cancelled_ts", "timestamp")),
        _t("inventory_snapshots", ("store_id", "product_id", "snapshot_ts"),
           ("snapshot_id", "string"), ("store_id", "string"), ("product_id", "string"),
           ("stock_quantity", "int"), ("reorder_level", "int"), ("snapshot_ts", "timestamp")),
        _t("inventory_events", ("event_id",),
           ("event_id", "string"), ("store_id", "string"), ("product_id", "string"),
           ("event_type", "string"), ("quantity", "int"), ("event_ts", "timestamp")),
        _t("application_logs", ("log_id",),
           ("log_id", "string"), ("event_ts", "timestamp"), ("service", "string"),
           ("log_level", "string"), ("message", "string")),
        _t("promotions", ("promotion_id",),
           ("promotion_id", "string"), ("name", "string"), ("category_id", "string"),
           ("discount_pct", "decimal"), ("start_date", "date"), ("end_date", "date"), tier=2),
        _t("order_promotions", ("order_id", "promotion_id"),
           ("order_id", "string"), ("promotion_id", "string"), ("discount_amount", "decimal"), tier=2),
        _t("returns_refunds", ("refund_id",),
           ("refund_id", "string"), ("payment_id", "string"), ("order_id", "string"),
           ("amount", "decimal"), ("reason", "string"), ("status", "string"),
           ("refund_ts", "timestamp"), tier=2),
        _t("reviews", ("review_id",),
           ("review_id", "string"), ("customer_id", "string"), ("product_id", "string"),
           ("order_id", "string"), ("rating", "int"), ("comment", "string", True),
           ("review_ts", "timestamp"), tier=2),
        _t("application_events", ("event_id",),
           ("event_id", "string"), ("customer_id", "string", True), ("session_id", "string"),
           ("event_name", "string"), ("event_ts", "timestamp"), ("metadata", "string"), tier=2),
        _t("weather", ("city_id", "observation_ts"),
           ("city_id", "string"), ("observation_ts", "timestamp"), ("temperature_c", "double"),
           ("rainfall_mm", "double"), ("humidity_pct", "double"), tier=2),
    ]
}

# Datasets that never change after the initial load; delivered once in the historical slice.
DIMENSION_DATASETS = ("categories", "products", "stores", "delivery_partners", "promotions")

# Datasets delivered as micro-batches in the stream slice (others stay daily batch files).
STREAM_DATASETS = ("orders", "order_items", "payments", "inventory_events", "application_events")

# Column whose IST date/hour decides the landing file; used for Bronze's late-arrival flag.
EVENT_TIME_COLUMNS = {
    "orders": "order_ts", "payments": "payment_ts", "deliveries": "pickup_ts", "cancellations": "cancelled_ts",
    "inventory_snapshots": "snapshot_ts", "inventory_events": "event_ts", "application_logs": "event_ts",
    "returns_refunds": "refund_ts", "reviews": "review_ts", "application_events": "event_ts",
    "weather": "observation_ts",
}

# Logical (simulated) system each dataset comes from.
SOURCE_SYSTEMS = {
    "categories": "catalog", "products": "catalog", "promotions": "catalog",
    "stores": "ops", "delivery_partners": "ops", "customers": "crm",
    "orders": "oms", "order_items": "oms", "order_promotions": "oms", "payments": "payments",
    "returns_refunds": "payments", "deliveries": "logistics", "cancellations": "oms",
    "inventory_snapshots": "wms", "inventory_events": "wms", "reviews": "app", "application_events": "app",
    "application_logs": "app_logs", "weather": "weather_api",
}

CORRUPT_RECORD = "_corrupt_record"
BRONZE_METADATA_COLUMNS = (
    "_batch_id", "_pipeline_run_id", "_source_file", "_load_type", "_source_system", "_schema_version",
    "_ingestion_ts", "_ingestion_date", "_record_hash", "_is_late", "_beyond_watermark",
)
SCHEMA_VERSION = "1"


def bronze_columns(dataset: str) -> list[str]:
    """Bronze column order: source columns (as received), corrupt-row capture, then metadata."""
    return SOURCE_SCHEMAS[dataset].names + [CORRUPT_RECORD, *BRONZE_METADATA_COLUMNS]
