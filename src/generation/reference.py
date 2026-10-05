"""Reference (dimension) data: categories, products, stores, delivery partners, promotions.

Brands are fictional. Product popularity follows a Zipf distribution; the most
popular products are the store's "focus SKUs" whose inventory is tracked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

from src.generation.context import GenContext, make_ids

# (category name, item nouns, (min price, max price))
CATEGORY_CATALOG: list[tuple[str, list[str], tuple[int, int]]] = [
    ("Dairy & Eggs", ["Toned Milk", "Curd", "Paneer", "Butter", "Cheese Slices", "Eggs", "Ghee", "Buttermilk"], (25, 550)),
    ("Bakery", ["Whole Wheat Bread", "Pav", "Croissant", "Muffin", "Rusk", "Garlic Bread"], (30, 180)),
    ("Fruits", ["Banana", "Apple", "Pomegranate", "Papaya", "Grapes", "Orange", "Mango"], (40, 250)),
    ("Vegetables", ["Onion", "Tomato", "Potato", "Spinach", "Capsicum", "Cucumber", "Coriander", "Carrot"], (15, 120)),
    ("Atta, Rice & Dal", ["Whole Wheat Atta", "Basmati Rice", "Toor Dal", "Moong Dal", "Poha", "Besan"], (60, 650)),
    ("Snacks & Namkeen", ["Potato Chips", "Bhujia", "Nachos", "Roasted Peanuts", "Popcorn", "Khakhra"], (20, 150)),
    ("Beverages", ["Cola", "Orange Juice", "Cold Coffee", "Coconut Water", "Iced Tea", "Energy Drink", "Soda"], (20, 180)),
    ("Ice Cream & Desserts", ["Vanilla Tub", "Chocolate Cone", "Kulfi", "Brownie", "Gulab Jamun Tin"], (30, 400)),
    ("Instant & Frozen Food", ["Instant Noodles", "Frozen Peas", "Frozen Paratha", "Ready-to-Eat Biryani", "Frozen Fries", "Pasta"], (15, 300)),
    ("Breakfast & Cereals", ["Corn Flakes", "Muesli", "Rolled Oats", "Peanut Butter", "Mixed Fruit Jam", "Honey"], (90, 500)),
    ("Tea & Coffee", ["Assam Tea", "Green Tea", "Instant Coffee", "Filter Coffee"], (80, 600)),
    ("Sweets & Chocolates", ["Milk Chocolate", "Dark Chocolate", "Toffees", "Kaju Katli", "Wafer Bar"], (10, 450)),
    ("Personal Care", ["Shampoo", "Body Wash", "Toothpaste", "Face Wash", "Deodorant", "Hand Wash"], (60, 450)),
    ("Household Cleaning", ["Dishwash Liquid", "Detergent Powder", "Floor Cleaner", "Toilet Cleaner", "Garbage Bags"], (50, 500)),
    ("Baby Care", ["Diapers", "Baby Wipes", "Baby Lotion", "Baby Food"], (150, 900)),
    ("Pet Care", ["Dog Food", "Cat Food", "Pet Treats"], (150, 900)),
    ("Gourmet & Organic", ["Organic Quinoa", "Extra Virgin Olive Oil", "Almonds", "Cashews", "Organic Honey"], (250, 1200)),
    ("Meat & Seafood", ["Chicken Breast", "Mutton Curry Cut", "Prawns", "Fish Fillet", "Chicken Sausages"], (180, 900)),
    ("Health & Wellness", ["Protein Bar", "Multivitamin", "Electrolyte Drink", "Chyawanprash"], (40, 800)),
    ("Stationery & Party", ["Notebook", "Gel Pens", "Candles", "Balloons", "Gift Wrap"], (20, 250)),
    ("Masala & Spices", ["Turmeric Powder", "Garam Masala", "Red Chilli Powder", "Cumin Seeds"], (30, 300)),
    ("Oils", ["Sunflower Oil", "Mustard Oil", "Groundnut Oil", "Rice Bran Oil"], (120, 900)),
    ("Sauces & Spreads", ["Tomato Ketchup", "Mayonnaise", "Chilli Sauce", "Chocolate Spread"], (60, 400)),
    ("Pooja Essentials", ["Agarbatti", "Camphor", "Diya", "Cotton Wicks"], (20, 200)),
    ("Kitchen & Home", ["Aluminium Foil", "Cling Wrap", "Paper Napkins", "Kitchen Towels"], (40, 300)),
]

BRANDS = [
    "FreshFarm", "DesiHarvest", "UrbanPantry", "GreenLeaf", "DailyDelight", "NutriKart", "PureNest",
    "SunMeadow", "KhetSe", "HappyBite", "Annapurna Foods", "BlueRiver", "Swadeshi Select", "Crunchies",
    "MorningGlow", "SparkleHome", "TinyToes", "PawPals", "Organica", "SeaCatch", "VitaBoost",
    "PaperTrail", "SpiceRoute", "GoldenDrop", "SaucyChef", "Shubh Pooja", "HomeEase", "ChillZone",
]

PACK_SIZES = ["100 g", "200 g", "500 g", "1 kg", "200 ml", "500 ml", "1 L", "Pack of 2", "Pack of 4", "Pack of 6"]

LOCALITIES = {
    "C01": ["Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "Jayanagar", "Malleshwaram",
            "Electronic City", "Hebbal", "BTM Layout", "Marathahalli"],
    "C02": ["Andheri West", "Bandra", "Powai", "Lower Parel", "Malad", "Goregaon", "Chembur",
            "Dadar", "Thane West", "Vashi"],
    "C03": ["Saket", "Dwarka", "Rohini", "Lajpat Nagar", "Karol Bagh", "Vasant Kunj", "Janakpuri",
            "Mayur Vihar", "Pitampura", "Hauz Khas"],
}

PROMO_THEMES = ["Monsoon Saver", "Weekend Bonanza", "Flash Deal", "Mega Savings", "Festive Treat", "Super Value"]


@dataclass
class ReferenceData:
    categories: pd.DataFrame
    products: pd.DataFrame       # + category_idx, popularity_rank, popularity_weight
    stores: pd.DataFrame         # + city_idx, demand_factor
    partners: pd.DataFrame       # + store_idx
    promotions: pd.DataFrame     # + category_idx (start/end as datetime64[D])
    focus_products: np.ndarray   # product indices whose stock is tracked in every store

    @property
    def n_products(self) -> int:
        return len(self.products)


def _price(rng: np.random.Generator, lo: int, hi: int) -> float:
    value = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    value = max(lo, round(value))
    if value > 50 and rng.random() < 0.5:
        value = value - (value % 10) + 9  # "...9" price endings
    return float(value)


def build_reference(ctx: GenContext) -> ReferenceData:
    cfg = ctx.cfg
    rng = ctx.rng("reference")
    fake = Faker("en_IN")
    fake.seed_instance(ctx.faker_seed("reference"))

    # Categories
    n_cat = cfg.counts.categories
    if n_cat > len(CATEGORY_CATALOG):
        raise ValueError(f"at most {len(CATEGORY_CATALOG)} categories are supported")
    catalog = CATEGORY_CATALOG[:n_cat]
    categories = pd.DataFrame({
        "category_id": make_ids("CAT", n_cat, 2),
        "category_name": [c[0] for c in catalog],
    })

    # Products: spread evenly across categories, Zipf popularity over a random ranking
    n_prod = cfg.counts.products
    cat_idx = rng.permutation(np.arange(n_prod) % n_cat)
    brands_by_cat = [rng.choice(BRANDS, size=4, replace=False) for _ in range(n_cat)]
    names, brands, prices = [], [], []
    for ci in cat_idx:
        _, nouns, (lo, hi) = catalog[ci]
        brand = str(rng.choice(brands_by_cat[ci]))
        names.append(f"{brand} {rng.choice(nouns)} {rng.choice(PACK_SIZES)}")
        brands.append(brand)
        prices.append(_price(rng, lo, hi))
    rank = rng.permutation(n_prod) + 1
    products = pd.DataFrame({
        "product_id": make_ids("P", n_prod, 4),
        "product_name": names,
        "category_id": categories["category_id"].to_numpy()[cat_idx],
        "brand": brands,
        "price": prices,
        "category_idx": cat_idx,
        "popularity_rank": rank,
        "popularity_weight": rank.astype(float) ** -cfg.demand.product_popularity_zipf_s,
    })
    focus = np.argsort(rank, kind="stable")[: cfg.focus_skus]

    # Stores: stores_per_city around each city centre
    rows = []
    for city_idx, city in enumerate(cfg.cities):
        localities = LOCALITIES.get(city.city_id, [f"Zone {i + 1}" for i in range(20)])
        for j in range(cfg.counts.stores_per_city):
            radius = cfg.store_radius_km * np.sqrt(rng.random())
            angle = rng.uniform(0, 2 * np.pi)
            rows.append({
                "store_name": f"QC Dark Store {localities[j % len(localities)]}",
                "city_id": city.city_id, "city": city.city, "state": city.state,
                "latitude": round(city.lat + radius / 111.0 * np.sin(angle), 5),
                "longitude": round(city.lon + radius / (111.0 * np.cos(np.radians(city.lat))) * np.cos(angle), 5),
                "city_idx": city_idx,
            })
    stores = pd.DataFrame(rows)
    stores.insert(0, "store_id", make_ids("S", len(stores), 2))
    factor = rng.lognormal(0.0, 0.25, size=len(stores))
    stores["demand_factor"] = factor / factor.mean()

    # Delivery partners: assigned round-robin to stores
    n_partners = len(stores) * cfg.ratios.partners_per_store
    store_idx = np.arange(n_partners) % len(stores)
    partners = pd.DataFrame({
        "partner_id": make_ids("DP", n_partners, 4),
        "partner_name": [fake.name() for _ in range(n_partners)],
        "store_id": stores["store_id"].to_numpy()[store_idx],
        "city_id": stores["city_id"].to_numpy()[store_idx],
        "availability_status": rng.choice(["active", "on_leave", "inactive"], size=n_partners, p=[0.85, 0.10, 0.05]),
        "store_idx": store_idx,
    })
    # Every store keeps at least one active partner
    for s in range(len(stores)):
        mask = (partners["store_idx"] == s).to_numpy()
        if not (partners.loc[mask, "availability_status"] == "active").any():
            partners.loc[np.flatnonzero(mask)[0], "availability_status"] = "active"

    # Promotions: category discounts with 3-14 day windows inside the period
    n_promo = cfg.counts.promotions
    cat_pop = np.bincount(cat_idx, weights=products["popularity_weight"].to_numpy(), minlength=n_cat)
    promo_cat = rng.choice(n_cat, size=n_promo, p=cat_pop / cat_pop.sum())
    duration = np.minimum(rng.integers(3, 15, size=n_promo), ctx.n_days)
    start_off = rng.integers(0, ctx.n_days - duration + 1)
    discount = rng.choice([5, 10, 15, 20, 25, 30], size=n_promo)
    promotions = pd.DataFrame({
        "promotion_id": make_ids("PR", n_promo, 3),
        "name": [f"{catalog[c][0]} {d}% Off - {rng.choice(PROMO_THEMES)}" for c, d in zip(promo_cat, discount)],
        "category_id": categories["category_id"].to_numpy()[promo_cat],
        "discount_pct": discount.astype(float),
        "start_date": ctx.days[start_off],
        "end_date": ctx.days[start_off + duration - 1],
        "category_idx": promo_cat,
    })

    return ReferenceData(categories, products, stores, partners, promotions, focus)


def active_promotions_matrix(ctx: GenContext, ref: ReferenceData) -> np.ndarray:
    """Boolean [day, category] matrix: is any promotion active for the category that day."""
    n_cat = len(ref.categories)
    active = np.zeros((ctx.n_days, n_cat), dtype=bool)
    start = ((ref.promotions["start_date"].to_numpy() - ctx.days[0]) // np.timedelta64(1, "D")).astype(int)
    end = ((ref.promotions["end_date"].to_numpy() - ctx.days[0]) // np.timedelta64(1, "D")).astype(int)
    for s, e, c in zip(start, end, ref.promotions["category_idx"].to_numpy()):
        active[s : e + 1, c] = True
    return active
