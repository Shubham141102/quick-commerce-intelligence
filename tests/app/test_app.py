"""Workspaces (Phases 4D, 5B, 5C, 5E): publish snapshot, role checks in data functions, login, pages render."""

import bcrypt
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.common.paths import PROJECT_ROOT  # noqa: E402
from src.ml.tables import ML_TABLES  # noqa: E402
from src.orchestration.publish import DIMENSIONS  # noqa: E402
from src.serving import queries as q  # noqa: E402
from src.serving.db import get_snapshot  # noqa: E402
from src.serving.permissions import AccessDenied  # noqa: E402
from src.transformations.gold.run import GOLD_TABLES  # noqa: E402

INVENTORY_USER = {"username": "inventory", "name": "Inventory Manager (demo)", "role": "inventory_manager"}
SECRETS = {"users": {"inventory": {"name": "Inventory Manager (demo)", "role": "inventory_manager",
                                   "password_hash": bcrypt.hashpw(b"pw-inv", bcrypt.gensalt(4)).decode()}}}
INVENTORY_FUNCTIONS = [
    (q.inventory_kpis, ()), (q.stock_health_by_store, ()),
    (q.forecast_accuracy, ()), (q.accuracy_by_category, ()), (q.feature_importance, ()),
    (q.risk_list, ()), (q.replenishment_assumptions, ()), (q.stockout_backtest, ()),
    (q.lost_sales_by_store, ()), (q.top_lost_sales_skus, (5,)),
]


def test_snapshot_published_within_limits(app_env):
    result = app_env["publish_result"].results["publish"]
    assert result["tables"] == len(GOLD_TABLES) + len(ML_TABLES) + len(DIMENSIONS) + 1 and result["size_mb"] < 50
    chunks = get_snapshot().query("SELECT doc_type, count(*) AS n FROM rag_chunks GROUP BY ALL")
    assert set(chunks["doc_type"]) == {"policy", "metric", "method"}   # + 1 = rag_chunks (Phase 6A)
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
    assert "High risk" in {m.label for m in at.metric}      # landed directly in the Inventory workspace


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


BUSINESS_USER = {"username": "business", "name": "Business Analyst (demo)", "role": "business_analyst"}


def _business_functions():
    b = q.period_bounds("business_analyst")
    s, e = b["first_day"].date(), b["last_day"].date()
    return [(q.sales_kpis, (s, e)), (q.sales_trend, (s, e)), (q.revenue_by_store, (s, e)),
            (q.category_contribution, (s, e)), (q.top_products, (s, e)), (q.delivery_kpis, (s, e)),
            (q.delivery_trend, (s, e)), (q.delivery_by_store, (s, e)), (q.cancellations_by_reason, (s, e)),
            (q.anomaly_summary, ()), (q.anomaly_list, ()), (q.anomaly_series, ("A0001",))]


def test_business_functions_check_the_role(app_env):
    for fn, args in _business_functions():
        assert fn("business_analyst", *args) is not None
        assert fn("admin", *args) is not None
        for role in ("inventory_manager", "marketing_manager", "data_engineer", None):
            with pytest.raises(AccessDenied):
                fn(role, *args)


def test_shared_filters_open_to_every_business_persona(app_env):
    for role in ("inventory_manager", "business_analyst", "marketing_manager", "admin"):
        assert len(q.stores(role)) > 0 and len(q.categories(role)) > 0
    with pytest.raises(AccessDenied):
        q.stores("data_engineer")


def test_business_numbers_match_gold(app_env):
    b = q.period_bounds("business_analyst")
    s, e = b["first_day"].date(), b["last_day"].date()
    total = q.sales_kpis("business_analyst", s, e)["current"]
    by_store = q.revenue_by_store("business_analyst", s, e)
    assert abs(float(by_store["net_revenue"].sum()) - float(total["net_revenue"])) < 0.01
    assert int(by_store["orders_completed"].sum()) == int(total["orders_completed"])


def test_business_workspace_renders(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "business.py"), default_timeout=120)
    at.session_state["user"] = BUSINESS_USER
    at.run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}
    assert {"Net revenue", "Completed orders", "Average order value", "On-time rate"} <= labels
    assert not at.error


def test_inventory_role_is_blocked_from_business(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "business.py"), default_timeout=120)
    at.session_state["user"] = INVENTORY_USER
    at.run()
    assert not at.exception and any("cannot open" in e.value for e in at.error) and not at.metric


def test_anomaly_list_and_series_agree(app_env):
    found = q.anomaly_list("business_analyst")
    assert len(found) > 0 and found["score"].is_monotonic_decreasing
    summary = q.anomaly_summary("business_analyst")
    assert int(summary["anomalies"].sum()) == len(found)
    series = q.anomaly_series("business_analyst", found["anomaly_id"].iloc[0])
    assert len(series) > 0 and (series["expected"] >= 0).all()


# ------------------------------------------------------------------------------------------- marketing (5E)
MARKETING_USER = {"username": "marketing", "name": "Marketing Manager (demo)", "role": "marketing_manager"}


def _marketing_functions():
    seg = int(q.segment_profiles("marketing_manager")["segment_id"].iloc[0])
    cust = q.recommendation_customers("marketing_manager", None, 1)["customer_id"].iloc[0]
    prod = q.rule_products("marketing_manager")["product_id"].iloc[0]
    return [(q.marketing_kpis, ()), (q.segment_profiles, ()), (q.segmentation_selection, ()),
            (q.segment_customers, (seg,)), (q.basket_rules, ()), (q.rule_products, ()), (q.product_partners, (prod,)),
            (q.recommendation_metrics, ()), (q.recommendation_customers, ()), (q.customer_recommendations, (cust,)),
            (q.segment_top_recommendations, (seg,)), (q.retention_cohorts, ()), (q.retention_status, ()),
            (q.retention_customers, ()), (q.promotion_metrics, ())]


def test_marketing_functions_check_the_role(app_env):
    for fn, args in _marketing_functions():
        assert fn("marketing_manager", *args) is not None
        assert fn("admin", *args) is not None
        for role in ("inventory_manager", "business_analyst", "data_engineer", None):
            with pytest.raises(AccessDenied):
                fn(role, *args)


def test_marketing_numbers_are_consistent(app_env):
    role = "marketing_manager"
    prof = q.segment_profiles(role)
    k = q.marketing_kpis(role)
    assert int(prof["customers"].sum()) == int(k["buyers"])            # every buyer is in exactly one segment
    assert abs(float(prof["share"].sum()) - 1) < 0.01 and prof["segment_label"].is_unique
    assert int(q.retention_status(role)["customers"].sum()) == int(k["customers"])
    recs = q.customer_recommendations(role, q.recommendation_customers(role, None, 1)["customer_id"].iloc[0])
    assert list(recs["rank"]) == list(range(1, len(recs) + 1)) and len(recs) <= 10
    assert (q.basket_rules(role)["lift"] >= 2).all()
    assert set(q.recommendation_metrics(role)["method"]) == {"hybrid", "repeat", "popularity"}


def test_marketing_workspace_renders(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "marketing.py"), default_timeout=120)
    at.session_state["user"] = MARKETING_USER
    at.run()
    assert not at.exception, at.exception
    labels = {m.label for m in at.metric}
    assert {"Customers", "Repeat rate", "Hybrid (used)", "Promotions"} <= labels
    assert not at.error


def test_business_role_is_blocked_from_marketing(app_env):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "marketing.py"), default_timeout=120)
    at.session_state["user"] = BUSINESS_USER
    at.run()
    assert not at.exception and any("cannot open" in e.value for e in at.error) and not at.metric


# ------------------------------------------------------------------------------------------- landing (frontend)
def _home(user):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "Home.py"), default_timeout=120)
    if user:
        at.session_state["user"] = user
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("user, expected", [
    (INVENTORY_USER, "High risk"),
    (BUSINESS_USER, "Net revenue"),
    ({"username": "marketing", "name": "Marketing Manager (demo)", "role": "marketing_manager"}, "Repeat rate"),
])
def test_each_persona_lands_in_its_own_workspace(app_env, user, expected):
    at = _home(user)
    assert expected in {m.label for m in at.metric} and not at.error


def test_signed_out_visitor_sees_only_the_login_page(app_env):
    at = _home(None)
    assert len(at.text_input) == 2 and not at.metric


def test_admin_lands_on_overview_and_engineer_on_hold_page(app_env):
    admin = _home({"username": "admin", "name": "Admin (demo)", "role": "admin"})
    assert "Tables in snapshot" in {m.label for m in admin.metric}
    engineer = _home({"username": "engineer", "name": "Data Engineer (demo)", "role": "data_engineer"})
    assert any("on hold" in i.value for i in engineer.info) and not engineer.error


# ------------------------------------------------------------------------------------------- assistant (6E)
def _assistant_page(user):
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "assistant.py"), default_timeout=120)
    at.session_state["user"] = user
    at.run()
    assert not at.exception, at.exception
    return at


def test_assistant_page_answers_with_badge_and_sources(app_env):
    at = _assistant_page(INVENTORY_USER)
    assert len(at.button) >= 3                                         # suggested questions
    at.chat_input[0].set_value("Which SKUs are at high risk and what does the policy say?").run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Data + policy" in text and "High risk" in text and "POL-INV §" in text
    assert len(at.dataframe) >= 1 and any(e.label == "Evidence & sources" for e in at.expander)


def test_assistant_page_refuses_other_workspace_data(app_env):
    at = _assistant_page(INVENTORY_USER)
    at.chat_input[0].set_value("What was net revenue last week?").run()
    text = " ".join(m.value for m in at.markdown)
    assert "Not in your workspace" in text and "Business & Revenue" in text and "₹" not in text


def test_assistant_suggested_question_click(app_env):
    at = _assistant_page(BUSINESS_USER)
    at.button[0].click().run()
    assert not at.exception and len(at.chat_message) == 2               # question + answer
