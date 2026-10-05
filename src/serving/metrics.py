"""Shared metric definitions (Project_Plan_v2.md §7.2) — the one place business metrics are defined.

Gold tables compute metrics with the Column helpers below; dashboards and the RAG assistant read
the resulting Gold columns and never re-derive them. `METRICS` is rendered into
docs/metric_definitions.md by `python -m scripts.build_docs`.

Decisions (agreed 2026-10-04): line revenue uses the catalog price when a line's unit price was
flagged as a mismatch; only completed orders count toward revenue; only completed refunds are
subtracted, on the refund's date.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import Column
from pyspark.sql import functions as F

BULK_ORDER_QUANTITY = 8


@dataclass(frozen=True)
class Metric:
    name: str
    definition: str
    formula: str


METRICS: tuple[Metric, ...] = (
    Metric("Completed order", "An order that was delivered, has a successful payment and was not cancelled.",
           "`slv_orders.is_completed`"),
    Metric("Line revenue", "Revenue of one order line. Uses the catalog price when the line's unit price was flagged "
           "(`dq_unit_price_mismatch`), otherwise the recorded line amount.",
           "quantity × (catalog price if flagged else unit_price)"),
    Metric("GMV", "Gross merchandise value of completed orders, before discounts.", "Σ line revenue (completed orders)"),
    Metric("Discount", "Promotion discounts on completed orders.", "Σ order_promotions.discount_amount (completed orders)"),
    Metric("Refunds", "Refunds with status `completed`, counted on the refund's business date.",
           "Σ returns_refunds.amount where status = completed"),
    Metric("Net revenue", "What the business keeps.", "GMV − Discount − Refunds"),
    Metric("Order value", "Value of a completed order at checkout.", "order GMV − order discount"),
    Metric("Average order value (AOV)", "Average checkout value of completed orders. Refunds are not included "
           "because they are dated by refund date, not order date.", "(GMV − Discount) ÷ completed orders"),
    Metric("Units", "Items sold in completed orders.", "Σ quantity (completed orders)"),
    Metric("Orders placed", "Every order in Silver, whatever its outcome.", "count(orders)"),
    Metric("Cancellation rate", "Share of placed orders that were cancelled.", "cancelled orders ÷ orders placed"),
    Metric("On-time delivery rate", "Share of delivered orders delivered within the SLA (15 minutes from pickup).",
           "delivered with delivery_minutes ≤ 15 ÷ delivered"),
    Metric("Units sold (inventory)", "Units leaving stock: every order line of a focus SKU except orders cancelled "
           "before dispatch.", "Σ quantity (orders not pre-dispatch cancelled)"),
    Metric("Days of inventory", "How many days the closing stock lasts at the recent sales rate. Empty when there "
           "were no sales in the last 14 days (never infinite).", "closing stock ÷ average daily units sold (last 14 days)"),
    Metric("Stockout day", "A day on which a focus SKU's stock reached zero at any point.",
           "min stock during the day ≤ 0"),
    Metric("Bulk order", f"An order with any line of {BULK_ORDER_QUANTITY} or more units (normal lines are 1–5).",
           f"max(quantity) ≥ {BULK_ORDER_QUANTITY}"),
    Metric("Support / confidence / lift", "Basket association between products A and B over completed orders.",
           "support = baskets(A∧B) ÷ baskets; confidence(A→B) = baskets(A∧B) ÷ baskets(A); "
           "lift = confidence(A→B) ÷ (baskets(B) ÷ baskets)"),
)


def line_revenue() -> Column:
    """Trusted line revenue (needs columns quantity, line_amount, catalog_price, dq_unit_price_mismatch)."""
    at_catalog = F.round(F.col("quantity") * F.col("catalog_price"), 2)
    return F.when(F.col("dq_unit_price_mismatch"), at_catalog).otherwise(F.col("line_amount")).cast("decimal(12,2)")


def ratio(numerator: Column, denominator: Column, digits: int = 4) -> Column:
    """numerator ÷ denominator, empty (not infinite) when the denominator is 0."""
    return F.when(denominator > 0, F.round(numerator / denominator, digits))


def money(c: Column) -> Column:
    return F.coalesce(c, F.lit(0)).cast("decimal(12,2)")
