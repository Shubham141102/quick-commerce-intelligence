"""Declarative Silver rules per dataset (Project_Plan_v2.md §6.3, decisions agreed on 2026-10-04).

Each spec lists normalisation, allowed values, ranges, foreign keys, extra rules,
lookups (helper columns joined from already-built Silver tables) and derived columns.
Required-value and type checks come automatically from `src/common/schemas.py`.

Decisions: strict cascade (children of rejected parents are quarantined); business rules
are soft flags except refund > payment; an invalid email is blanked and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.quality.rules import Rule, business_rule, range_rule, sequence_rule
from src.quality.standardize import business_date

if TYPE_CHECKING:
    from src.transformations.silver.engine import SilverContext


@dataclass(frozen=True)
class FK:
    column: str
    parent: str
    parent_column: str


@dataclass(frozen=True)
class SilverSpec:
    dataset: str
    lower: tuple[str, ...] = ()            # code values -> lowercase
    upper: tuple[str, ...] = ()            # code values -> uppercase
    initcap: tuple[str, ...] = ()          # names whose canonical form is Title Case
    majority: tuple[str, ...] = ()         # canonical spelling = most frequent variant
    email: tuple[str, ...] = ()            # lowercase; invalid format -> null + dq_email_invalid
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rules: tuple[Rule, ...] = ()
    fks: tuple[FK, ...] = ()
    lookups: Callable[[DataFrame, SilverContext], DataFrame] | None = None
    derive: Callable[[DataFrame, SilverContext], DataFrame] | None = None
    derived: tuple[tuple[str, str], ...] = ()   # (column, logical type) added by derive / post-processing

    def soft_rules(self) -> list[Rule]:
        return [r for r in self.rules if not r.hard]


CITY = "__cities__"   # placeholder replaced by the configured city ids
ORDER_STATUS = ("delivered", "cancelled")
PAYMENT_METHODS = ("upi", "card", "wallet", "cod")
PAYMENT_STATUS = ("success", "failed", "voided", "cancelled")
DELIVERY_STATUS = ("delivered", "cancelled", "failed")
CANCEL_STAGE = ("pre_dispatch", "post_dispatch")
CANCEL_REASONS = ("customer_changed_mind", "item_unavailable", "payment_issue", "duplicate_order",
                  "delivery_delayed", "customer_unreachable", "address_issue")
INVENTORY_EVENT_TYPES = ("restock", "damage", "adjustment")
PARTNER_AVAILABILITY = ("active", "on_leave", "inactive")
REFUND_REASONS = ("order_cancelled", "damaged_item", "missing_item", "quality_issue", "expired_product")
REFUND_STATUS = ("completed", "pending", "rejected")
APP_EVENTS = ("app_open", "search", "view_product", "add_to_cart", "view_cart", "checkout_start", "payment_page")
LOG_LEVELS = ("INFO", "WARN", "ERROR", "DEBUG")
SERVICES = ("order-service", "payment-service", "inventory-service", "delivery-service", "search-service")


def _with_business_date(ts: str) -> Callable[[DataFrame, SilverContext], DataFrame]:
    return lambda df, ctx: df.withColumn("business_date", business_date(ts))


def _lookup(df: DataFrame, parent: DataFrame, on: str, columns: dict[str, str], parent_on: str | None = None) -> DataFrame:
    """Left-join helper columns from an accepted Silver table (`columns`: parent column -> new name)."""
    pk = f"__lk_{on}"
    right = parent.select(F.col(parent_on or on).alias(pk), *[F.col(c).alias(n) for c, n in columns.items()]).dropDuplicates([pk])
    return df.join(F.broadcast(right), df[on] == right[pk], "left").drop(pk)


# ------------------------------------------------------------------------------------- lookups

def _order_items_lookups(df, ctx):
    return _lookup(df, ctx.accepted["products"], "product_id", {"price": "__catalog_price"})


def _order_promotions_lookups(df, ctx):
    df = _lookup(df, ctx.accepted["orders"], "order_id", {"business_date": "__order_date"})
    return _lookup(df, ctx.accepted["promotions"], "promotion_id",
                   {"start_date": "__promo_start", "end_date": "__promo_end"})


def _payments_lookups(df, ctx):
    df = _lookup(df, ctx.order_totals(), "order_id", {"computed_total": "__computed_total"})
    return _lookup(df, ctx.accepted["orders"], "order_id", {"total_amount": "__order_total"})


def _payment_amount_mismatch():
    # An order's stored total can be wrong (tampered total) or its lines incomplete (a line lost
    # to a bad order_id); the payment matches one of the two. Flag only if it matches neither.
    off_total = F.abs(F.col("amount") - F.col("__order_total")) > 0.005
    off_computed = F.coalesce(F.abs(F.col("amount") - F.col("__computed_total")) > 0.005, F.lit(True))
    return off_total & off_computed


def _deliveries_lookups(df, ctx):
    df = _lookup(df, ctx.accepted["delivery_partners"], "partner_id", {"city_id": "__partner_city"})
    stores = ctx.accepted["orders"].join(ctx.accepted["stores"].select("store_id", F.col("city_id").alias("__store_city")),
                                         "store_id").select("order_id", "__store_city")
    return _lookup(df, stores, "order_id", {"__store_city": "__store_city"})


def _cancellations_lookups(df, ctx):
    return _lookup(df, ctx.accepted["orders"], "order_id", {"order_ts": "__order_ts"})


def _reviews_lookups(df, ctx):
    bought = (ctx.deduped["order_items"].select(F.concat_ws("|", "order_id", "product_id").alias("__purchase_key"))
              .distinct().withColumn("__purchased", F.lit(True)))
    incomplete = (ctx.rejected["order_items"].where(F.col("order_id").isNotNull()).select("order_id").distinct()
                  .withColumn("__items_incomplete", F.lit(True)))
    df = df.withColumn("__purchase_key", F.concat_ws("|", "order_id", "product_id"))
    df = df.join(F.broadcast(bought), "__purchase_key", "left").drop("__purchase_key")
    return _lookup(df, incomplete, "order_id", {"__items_incomplete": "__items_incomplete"})


def _unverified_purchase():
    # null ("could not verify") when some of the order's item rows were rejected
    return F.when(F.col("__items_incomplete").isNull(), F.col("__purchased").isNull())


def _refunds_lookups(df, ctx):
    return _lookup(df, ctx.accepted["payments"], "payment_id", {"amount": "__payment_amount", "payment_ts": "__payment_ts"})


# ------------------------------------------------------------------------------------- derived

def _order_items_derive(df, ctx):
    return df.withColumn("line_amount", F.round(F.col("quantity") * F.col("unit_price"), 2).cast("decimal(12,2)"))


def _deliveries_derive(df, ctx):
    minutes = (F.unix_timestamp("delivered_ts") - F.unix_timestamp("pickup_ts")) / 60
    return (df.withColumn("business_date", business_date("pickup_ts"))
            .withColumn("delivery_minutes", F.when(F.col("status") == "delivered", F.round(minutes, 1))))


def _application_events_derive(df, ctx):
    return (df.withColumn("business_date", business_date("event_ts"))
            .withColumn("platform", F.get_json_object("metadata", "$.platform"))
            .withColumn("app_version", F.get_json_object("metadata", "$.app_version")))


def _weather_derive(df, ctx):
    return df.withColumn("business_date", business_date("observation_ts"))


SPECS: dict[str, SilverSpec] = {s.dataset: s for s in [
    SilverSpec("categories", initcap=("category_name",)),
    SilverSpec("stores", initcap=("city",), enums={"city_id": (CITY,)},
               rules=(range_rule("latitude", -90, 90), range_rule("longitude", -180, 180))),
    SilverSpec("customers", email=("email",), enums={"city_id": (CITY,)}, derived=(("dq_email_invalid", "boolean"),)),
    SilverSpec("weather", enums={"city_id": (CITY,)},
               rules=(range_rule("humidity_pct", 0, 100), range_rule("rainfall_mm", 0),
                      range_rule("temperature_c", -20, 60)),
               derive=_weather_derive, derived=(("business_date", "date"),)),
    SilverSpec("application_logs", upper=("log_level",), lower=("service",),
               enums={"log_level": LOG_LEVELS, "service": SERVICES},
               derive=_with_business_date("event_ts"), derived=(("business_date", "date"),)),
    SilverSpec("products", majority=("brand",), fks=(FK("category_id", "categories", "category_id"),),
               rules=(range_rule("price", 0, 100_000, lo_inclusive=False),)),
    SilverSpec("promotions", fks=(FK("category_id", "categories", "category_id"),),
               rules=(range_rule("discount_pct", 0, 100, lo_inclusive=False),
                      sequence_rule("end_before_start", "end_date", "start_date"))),
    SilverSpec("delivery_partners", lower=("availability_status",),
               enums={"availability_status": PARTNER_AVAILABILITY, "city_id": (CITY,)},
               fks=(FK("store_id", "stores", "store_id"),)),
    SilverSpec("orders", lower=("status",), enums={"status": ORDER_STATUS},
               fks=(FK("customer_id", "customers", "customer_id"), FK("store_id", "stores", "store_id")),
               rules=(range_rule("total_amount", 0),),
               derive=_with_business_date("order_ts"),
               # computed_total_amount, dq_total_mismatch, is_outlier and is_completed need child tables;
               # they are filled in by the orders post-processing step (run.py)
               derived=(("business_date", "date"), ("computed_total_amount", "decimal"),
                        ("dq_total_mismatch", "boolean"), ("is_outlier", "boolean"), ("is_completed", "boolean"))),
    SilverSpec("order_items",
               fks=(FK("order_id", "orders", "order_id"), FK("product_id", "products", "product_id")),
               rules=(range_rule("quantity", 1, 100), range_rule("unit_price", 0, lo_inclusive=False),
                      business_rule("unit_price_mismatch",
                                    lambda: F.abs(F.col("unit_price") - F.col("__catalog_price")) > 0.005,
                                    "`unit_price` should equal the product's catalog price (`slv_products.price`)")),
               lookups=_order_items_lookups, derive=_order_items_derive, derived=(("line_amount", "decimal"),)),
    SilverSpec("order_promotions",
               fks=(FK("order_id", "orders", "order_id"), FK("promotion_id", "promotions", "promotion_id")),
               rules=(range_rule("discount_amount", 0, lo_inclusive=False),
                      business_rule("promo_inactive", lambda: (F.col("__order_date") < F.col("__promo_start"))
                                    | (F.col("__order_date") > F.col("__promo_end")),
                                    "the promotion should be active on the order's business date (IST)")),
               lookups=_order_promotions_lookups),
    SilverSpec("payments", lower=("method", "status"), enums={"method": PAYMENT_METHODS, "status": PAYMENT_STATUS},
               fks=(FK("order_id", "orders", "order_id"),),
               rules=(range_rule("amount", 0), range_rule("attempt_no", 1, 5),
                      business_rule("amount_mismatch", _payment_amount_mismatch,
                                    "`amount` should equal the order's stored total or its total recomputed from "
                                    "its lines; flagged only when it matches neither")),
               lookups=_payments_lookups, derive=_with_business_date("payment_ts"), derived=(("business_date", "date"),)),
    SilverSpec("deliveries", lower=("status",), enums={"status": DELIVERY_STATUS},
               fks=(FK("order_id", "orders", "order_id"), FK("partner_id", "delivery_partners", "partner_id")),
               rules=(Rule("required_missing:delivered_ts",
                           lambda: (F.col("status") == "delivered") & F.col("__s_delivered_ts").isNull(),
                           description="`delivered_ts` is required when `status` is `delivered`"),
                      sequence_rule("delivered_before_pickup", "delivered_ts", "pickup_ts"),
                      business_rule("partner_wrong_city", lambda: F.col("__partner_city") != F.col("__store_city"),
                                    "the rider's city should match the city of the order's store")),
               lookups=_deliveries_lookups, derive=_deliveries_derive,
               derived=(("business_date", "date"), ("delivery_minutes", "double"))),
    SilverSpec("cancellations", lower=("stage", "reason"), enums={"stage": CANCEL_STAGE, "reason": CANCEL_REASONS},
               fks=(FK("order_id", "orders", "order_id"),),
               rules=(sequence_rule("cancelled_before_order", "cancelled_ts", "__order_ts",
                                    "`cancelled_ts` must not be before the order's `order_ts`"),),
               lookups=_cancellations_lookups, derive=_with_business_date("cancelled_ts"),
               derived=(("business_date", "date"),)),
    SilverSpec("inventory_snapshots",
               fks=(FK("store_id", "stores", "store_id"), FK("product_id", "products", "product_id")),
               rules=(range_rule("stock_quantity", 0), range_rule("reorder_level", 0)),
               derive=_with_business_date("snapshot_ts"), derived=(("business_date", "date"),)),
    SilverSpec("inventory_events", lower=("event_type",), enums={"event_type": INVENTORY_EVENT_TYPES},
               fks=(FK("store_id", "stores", "store_id"), FK("product_id", "products", "product_id")),
               rules=(Rule("range_invalid:quantity", lambda:
                           ((F.col("event_type") == "restock") & (F.col("quantity") <= 0))
                           | ((F.col("event_type") == "damage") & (F.col("quantity") >= 0))
                           | ((F.col("event_type") == "adjustment") & (F.col("quantity") == 0)),
                           description="`quantity` sign must match `event_type`: restock > 0, damage < 0, "
                                       "adjustment ≠ 0"),),
               derive=_with_business_date("event_ts"), derived=(("business_date", "date"),)),
    SilverSpec("reviews",
               fks=(FK("customer_id", "customers", "customer_id"), FK("product_id", "products", "product_id"),
                    FK("order_id", "orders", "order_id")),
               rules=(range_rule("rating", 1, 5),
                      business_rule("unverified_purchase", _unverified_purchase,
                                    "the reviewed product should be a line of the review's order; empty (could not "
                                    "verify) when some of that order's lines were rejected", unknown_is_pass=False)),
               lookups=_reviews_lookups, derive=_with_business_date("review_ts"), derived=(("business_date", "date"),)),
    SilverSpec("application_events", lower=("event_name",), enums={"event_name": APP_EVENTS},
               fks=(FK("customer_id", "customers", "customer_id"),),
               rules=(Rule("malformed:metadata", lambda: F.col("metadata").isNotNull()
                           & F.from_json("metadata", "map<string,string>").isNull(),
                           description="`metadata` must be a valid JSON object"),),
               derive=_application_events_derive,
               derived=(("business_date", "date"), ("platform", "string"), ("app_version", "string"))),
    SilverSpec("returns_refunds", lower=("reason", "status"), enums={"reason": REFUND_REASONS, "status": REFUND_STATUS},
               fks=(FK("payment_id", "payments", "payment_id"), FK("order_id", "orders", "order_id")),
               rules=(range_rule("amount", 0, lo_inclusive=False),
                      business_rule("refund_exceeds_payment", lambda: F.col("amount") > F.col("__payment_amount") + 0.005,
                                    "a refund cannot exceed the payment it refunds (financially impossible)", hard=True),
                      sequence_rule("refund_before_payment", "refund_ts", "__payment_ts",
                                    "`refund_ts` must not be before the refunded payment's `payment_ts`")),
               lookups=_refunds_lookups, derive=_with_business_date("refund_ts"), derived=(("business_date", "date"),)),
]}

# Processing order: every parent is built before its children.
SILVER_ORDER = (
    "categories", "stores", "customers", "weather", "application_logs",
    "products", "promotions", "delivery_partners",
    "orders",
    "order_items", "order_promotions", "payments", "deliveries", "cancellations",
    "inventory_snapshots", "inventory_events", "reviews", "application_events",
    "returns_refunds",
)
