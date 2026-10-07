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
    "business": {
        "icon": ":material/monitoring:", "code": "BIZ", "title": "Business & Revenue", "page": "views/business.py",
        "tagline": "How the business is performing: revenue, orders, delivery promise, and anything unusual.",
        "use_cases": ["Sales & revenue performance", "Delivery & operations", "Anomaly detection"],
    },
    "marketing": {
        "icon": ":material/campaign:", "code": "MKT", "title": "Customer Growth & Marketing", "page": "views/marketing.py",
        "tagline": "Know the customers, grow the basket and bring lapsing customers back.",
        "use_cases": ["Customer segmentation", "Basket & affinity", "Recommendations", "Retention",
                      "Promotion effectiveness"],
    },
    "engineering": {
        "icon": ":material/engineering:", "code": "ENG", "title": "Data Engineer", "page": "views/engineering.py",
        "tagline": "Pipeline runs, data quality, lineage and model monitoring.",
        "use_cases": ["Pipeline runs", "Data quality & quarantine", "Lineage & tables", "Model monitoring"],
        "on_hold": True,
    },
}
