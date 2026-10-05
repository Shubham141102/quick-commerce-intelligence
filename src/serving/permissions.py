"""Roles and workspaces (Project_Plan_v2.md §10.1). Checked inside every data function, not only in the menu.

Demo-grade access control: it separates personas for the demonstration; it is not production security.
"""

from __future__ import annotations

WORKSPACES = {
    "inventory": "Inventory & Supply Chain",
    "marketing": "Customer Growth & Marketing",
    "business": "Business & Revenue",
    "engineering": "Data Engineer",
}
ROLES = {
    "inventory_manager": {"inventory"},
    "marketing_manager": {"marketing"},
    "business_analyst": {"business"},
    "data_engineer": {"engineering"},
    "admin": set(WORKSPACES),
}
ROLE_LABELS = {
    "inventory_manager": "Inventory & Supply Chain Manager", "marketing_manager": "Growth & Marketing Manager",
    "business_analyst": "Business & Revenue Analyst", "data_engineer": "Data Engineer", "admin": "Demo administrator",
}


class AccessDenied(PermissionError):
    pass


def allowed_workspaces(role: str) -> list[str]:
    return [w for w in WORKSPACES if w in ROLES.get(role, set())]


def require(role: str | None, workspace: str) -> None:
    if not role or workspace not in ROLES.get(role, set()):
        raise AccessDenied(f"role {role!r} may not open the {WORKSPACES.get(workspace, workspace)} workspace")
