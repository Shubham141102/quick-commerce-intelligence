# Inventory Policy

Policy ID: POL-INV · Owner: Inventory & Supply Chain · Applies to: all dark stores · Version 1.0 (Apr 2025)

This policy governs stock of focus SKUs in the Quick-Commerce Intelligence demo network (synthetic data).

## 1. Scope

The policy covers the 25 focus SKUs of each dark store: the store's most popular products, which drive most orders and availability complaints. 12 stores × 25 focus SKUs = 300 store × SKU pairs. Other products are replenished centrally and are outside this policy. The goal is simple: keep focus SKUs on the shelf without over-stocking. A stockout on a focus SKU is treated as a service failure, because the shopper may cancel the order or buy elsewhere.

## 2. Stock risk tiers

Every focus SKU gets a risk tier each night, from closing stock, the reorder level and the demand forecast.

- **High risk:** stock is zero, or forecast demand over the next 3 days (supplier lead time of 2 days plus 1 day) is greater than or equal to the stock on hand. The SKU will very likely run out before a delivery can arrive.
- **Medium risk:** stock is at or below the reorder level, or less than 3 days of inventory remain.
- **Low risk:** none of the above.

High-risk SKUs must be reviewed the same day. Medium-risk SKUs must be reordered in the next nightly cycle.

## 3. Reorder rules

The reorder level of a SKU is set from its average daily demand: reorder level = 3 × average daily demand, rounded up, plus 1 unit. Stock is checked every night at 22:00 IST. When stock is at or below the reorder level and no delivery is already pending, a replenishment order is placed. Only one replenishment order may be open per store × SKU at a time. A SKU with fewer than 3 days of inventory (closing stock ÷ average daily units over the last 14 days) is reordered even if it is above its reorder level.

## 4. Order quantity

Stores replenish up to an order-up-to level: reorder level + 7 days of average demand + 2 units. When the forecast is available, the suggested order quantity is: forecast demand over the lead time and review period (2 + 1 days) + safety stock − current stock, rounded up and never negative. Safety stock = 1.65 × standard deviation of daily units over the last 28 days × √2, which targets a 95% service level. Suggested quantities are advisory: the store manager approves every order.

## 5. Supplier lead time

Suppliers deliver 1–2 days after an order is placed. Planning uses a 2-day lead time (the worst normal case). A delivery that has not arrived within 3 days is escalated to the supplier manager. During a known supplier outage, affected SKUs are marked High risk and substitutes are promoted in the app.

## 6. Stock counts and write-offs

Each store counts every focus SKU once a week, on Monday at 06:00 IST. The count is the source of truth: calculated stock is reset to the counted figure. A gap between counted and calculated stock above 2 units for the same SKU in two consecutive weeks must be investigated. Damaged or expired units are written off on the day they are found and recorded as damage events; adjustments after a count are recorded as adjustment events.

## 7. Stockouts and lost sales

A stockout day is any day on which a focus SKU's stock reached zero. Lost sales are estimated as the units the SKU normally sells on in-stock days (average of the previous 28 days) minus the units it actually sold, multiplied by the price. The estimate is reported before substitutes, so the store as a whole loses less. Stores with the highest lost sales are reviewed monthly.

## 8. Escalation

- More than 2 High-risk SKUs at one store at the end of a day: the store manager is notified.
- The same SKU out of stock on more than 3 days in a month: the reorder level is reviewed.
- A planned supplier outage: affected stores are told in advance and buffer stock is raised.
