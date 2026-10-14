"""Persona / workspace descriptions shared by the login page, page headers and the admin overview."""

from __future__ import annotations

# One entry per workspace: used by the login page, the page headers and the admin overview.
PERSONAS = {
    "inventory": {
        "icon": ":material/inventory_2:", "code": "INV", "title": "Inventory & Supply Chain", "page": "views/inventory.py",
        "tagline": "Keep shelves full without over-stocking: what will sell, what will run out, what to reorder.",
        "use_cases": ["Inventory health overview", "Demand forecasting", "Stockout risk & replenishment",
                      "Inventory explorer & lost sales"],
    },
    "growth": {
        "icon": ":material/monitoring:", "code": "GRO", "title": "Business & Growth", "page": "views/growth.py",
        "tagline": "Are we growing profitably and keeping customers? Revenue, the delivery promise, customers and "
                   "the levers to grow.",
        "use_cases": ["Performance (sales & delivery)", "Alerts (anomalies)", "Customers (segments & retention)",
                      "Growth levers (basket, recommendations, promotions)"],
    },
    "engineering": {
        "icon": ":material/engineering:", "code": "ENG", "title": "Data Engineer", "page": "views/engineering.py",
        "tagline": "Pipeline runs, data quality, lineage and model monitoring.",
        "use_cases": ["Pipeline runs", "Data quality & quarantine", "Lineage & tables", "Model monitoring"],
    },
}
