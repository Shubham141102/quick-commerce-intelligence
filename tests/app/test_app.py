"""Inventory workspace (Phase 4D): publish snapshot, role checks in data functions, login, pages render."""

import bcrypt
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.common.paths import PROJECT_ROOT  # noqa: E402
from src.serving import queries as q  # noqa: E402
from src.serving.permissions import AccessDenied  # noqa: E402

INVENTORY_USER = {"username": "inventory", "name": "Inventory Manager (demo)", "role": "inventory_manager"}
SECRETS = {"users": {"inventory": {"name": "Inventory Manager (demo)", "role": "inventory_manager",
                                   "password_hash": bcrypt.hashpw(b"pw-inv", bcrypt.gensalt(4)).decode()}}}
INVENTORY_FUNCTIONS = [
    (q.stores, ()), (q.categories, ()), (q.inventory_kpis, ()), (q.stock_health_by_store, ()),
    (q.forecast_accuracy, ()), (q.accuracy_by_category, ()), (q.feature_importance, ()),
    (q.risk_list, ()), (q.replenishment_assumptions, ()), (q.stockout_backtest, ()),
    (q.lost_sales_by_store, ()), (q.top_lost_sales_skus, (5,)),
]


def test_snapshot_published_within_limits(app_env):
    result = app_env["publish_result"].results["publish"]
    assert result["tables"] == 23 and result["size_mb"] < 50
    info = q.snapshot_info()
    assert info["pipeline_run_id"] == app_env["publish_result"].run_id


@pytest.mark.parametrize("fn,args", INVENTORY_FUNCTIONS)
def test_data_functions_check_the_role(app_env, fn, args):
    assert fn("inventory_manager", *args) is not None
    assert fn("admin", *args) is not None
    for role in ("marketing_manager", "business_analyst", "data_engineer", None):
        with pytest.raises(AccessDenied):
            fn(role, *args)


def test_login_with_demo_account(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=120)
    at.secrets["auth"] = SECRETS
    at.run()
    at.text_input[0].input("inventory")
    at.text_input[1].input("wrong")
    at.button[0].click().run()
    assert "user" not in at.session_state and at.error
    at.text_input[1].input("pw-inv")
    at.button[0].click().run()
    assert at.session_state["user"]["role"] == "inventory_manager"
    assert not at.exception


def test_inventory_workspace_renders(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "inventory.py"), default_timeout=120)
    at.session_state["user"] = INVENTORY_USER
    at.run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert {"High risk", "Medium risk", "SKUs to reorder", "Model WAPE"} <= set(labels)
    assert {"Estimated lost sales (all stores)", "Stockout days", "Est. lost sales"} <= set(labels)  # explorer tab
    assert len(at.dataframe) >= 4 and not at.error


def test_other_roles_are_blocked_from_inventory(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "inventory.py"), default_timeout=120)
    at.session_state["user"] = {"username": "m", "name": "M", "role": "marketing_manager"}
    at.run()
    assert not at.exception
    assert any("cannot open" in e.value for e in at.error)
    assert not at.metric
