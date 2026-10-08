"""Data tools of the business assistant (Phase 6C) against a real published snapshot (small run)."""

import pytest
from pydantic import ValidationError

from src.rag.entities import Lookups
from src.rag.tools import TOOLS, ToolInputError, available_tools, run_tool
from src.serving import queries as q
from src.serving.permissions import AccessDenied

ROLE_OF = {"inventory": "inventory_manager", "business": "business_analyst", "marketing": "marketing_manager"}


@pytest.fixture(scope="module")
def lk(app_env):
    return Lookups.from_snapshot()


def _params(name: str, lk: Lookups) -> dict:
    if name == "category_forecast":
        return {"category_id": lk.categories["category_id"].iloc[0]}
    if name == "bought_with":
        return {"product_id": q.rule_products("marketing_manager")["product_id"].iloc[0]}
    return {}


def test_each_role_sees_only_its_workspace_tools(app_env):
    for ws, role in ROLE_OF.items():
        assert {t.workspace for t in available_tools(role)} == {ws}
    assert len(available_tools("admin")) == len(TOOLS) and available_tools("data_engineer") == []


def test_every_tool_runs_for_its_role_with_source_and_no_customer_ids(app_env, lk):
    for name, tool in TOOLS.items():
        r = run_tool(name, ROLE_OF[tool.workspace], _params(name, lk), lk)
        assert r.headline and r.source.startswith("gld_") and r.as_of and r.scope, name
        assert len(r.table) <= 20, name
        # no customer identifiers (a "customers" count column is an aggregate, which is allowed)
        assert not [c for c in r.table.columns if "customer" in c.lower() and c.lower().endswith("id")], name


def test_tools_refuse_other_roles(app_env, lk):
    with pytest.raises(AccessDenied):
        run_tool("sales_overview", "inventory_manager", {}, lk)
    with pytest.raises(AccessDenied):
        run_tool("at_risk_skus", "marketing_manager", {}, lk)


def test_inputs_are_validated(app_env, lk):
    with pytest.raises(ToolInputError):
        run_tool("sales_overview", "business_analyst", {"start": lk.first_day.replace(year=2020)}, lk)
    with pytest.raises(ToolInputError):
        run_tool("at_risk_skus", "inventory_manager", {"store_ids": ["S99"]}, lk)
    with pytest.raises(ToolInputError):
        run_tool("category_forecast", "inventory_manager", {}, lk)          # needs a category
    with pytest.raises(ValidationError):
        run_tool("top_products", "business_analyst", {"limit": 500}, lk)     # at most 20 rows
    with pytest.raises(ValidationError):
        run_tool("at_risk_skus", "inventory_manager", {"sql": "DROP TABLE x"}, lk)   # no unknown parameters


def test_numbers_equal_the_dashboard_functions(app_env, lk):
    k = q.inventory_kpis("inventory_manager")
    r = run_tool("at_risk_skus", "inventory_manager", {}, lk)
    assert r.headline["High risk"] == str(int(k["high"])) and r.headline["Medium risk"] == str(int(k["medium"]))
    s = q.sales_kpis("business_analyst", lk.first_day, lk.last_day)["current"]
    r = run_tool("sales_overview", "business_analyst", {}, lk)
    assert r.headline["Net revenue"].startswith(f"₹{float(s['net_revenue']):,.0f}")
