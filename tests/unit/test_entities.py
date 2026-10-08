"""Entity extraction (Phase 6C) on hand-made lookups: stores, cities, categories, products, tiers, periods."""

from datetime import date

import pandas as pd

from src.rag.entities import Lookups, extract

LK = Lookups(
    stores=pd.DataFrame({"store_id": ["S01", "S05", "S06"],
                         "store_name": ["QC Dark Store Koramangala", "QC Dark Store Dwarka", "QC Dark Store Saket"],
                         "city": ["Bengaluru", "Delhi", "Delhi"]}),
    categories=pd.DataFrame({"category_id": ["C1", "C2"], "category_name": ["Dairy & Eggs", "Atta, Rice & Dal"]}),
    products=pd.DataFrame({"product_id": ["P1", "P2", "P3"],
                           "product_name": ["FreshFarm Baby Wipes 1 L", "FreshFarm Baby Wipes Pack of 6",
                                            "UrbanPantry Diapers 1 L"]}),
    first_day=date(2025, 4, 1), last_day=date(2025, 9, 30))


def test_store_tier_and_top_n():
    e = extract("Show the top 5 high risk SKUs in Dwarka", LK)
    assert e.store_ids == ["S05"] and e.tiers == ["High"] and e.limit == 5
    assert e.found["store"] == "Dwarka"


def test_city_means_all_its_stores_and_store_ids_work():
    assert extract("revenue in Delhi", LK).store_ids == ["S05", "S06"]
    assert extract("stock at S01", LK).store_ids == ["S01"]


def test_categories_by_full_name_or_part():
    assert extract("forecast for dairy next week", LK).category_ids == ["C1"]
    assert extract("Atta, Rice & Dal demand", LK).category_ids == ["C2"]


def test_product_longest_name_wins_and_sizes_ignored():
    e = extract("What is bought with FreshFarm Baby Wipes?", LK)
    assert e.product_ids in (["P1"], ["P2"]) and e.found["product"] == "FreshFarm Baby Wipes"
    assert extract("people who buy urbanpantry diapers", LK).product_ids == ["P3"]


def test_periods_relative_to_the_end_of_the_data():
    e = extract("revenue last week", LK)
    assert (e.start, e.end) == (date(2025, 9, 24), date(2025, 9, 30))
    e = extract("orders in the last 30 days", LK)
    assert (e.start, e.end) == (date(2025, 9, 1), date(2025, 9, 30))
    e = extract("sales in July", LK)
    assert (e.start, e.end, e.period_label) == (date(2025, 7, 1), date(2025, 7, 31), "July 2025")
    assert extract("May I see the stock?", LK).start is None               # "may" as a verb is not a month
    assert extract("revenue in May", LK).start == date(2025, 5, 1)
    assert extract("sales last month", LK).start == date(2025, 9, 1)
    assert extract("what is the reorder rule", LK).start is None             # no period mentioned


def test_detectors_and_no_false_tiers():
    assert extract("any payment failures or spikes?", LK).detectors == ["demand_spike", "payment_failure"]
    assert extract("is the forecast highly accurate", LK).tiers == []        # "high" not followed by risk / tier
