"""
tenant.py — Tenant/security domain scoping for PSA.

Each tenant is an isolated memory domain with its own storage paths.
The default tenant is "default" for single-user deployments.

Paths: ~/.psa/tenants/{tenant_id}/
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


def _validate_tenant_id(tenant_id: str):
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id {tenant_id!r}. "
            "Must be 1-64 lowercase alphanumeric characters, hyphens, or underscores."
        )


@dataclass
class Tenant:
    """A PSA tenant (isolated memory domain)."""

    tenant_id: str
    display_name: str
    created_at: str
    root_dir: str  # absolute path to tenant directory
    metadata: dict = field(default_factory=dict)

    @property
    def memory_db_path(self) -> str:
        return os.path.join(self.root_dir, "memory.sqlite3")

    @property
    def atlas_dir(self) -> str:
        return os.path.join(self.root_dir, "atlas")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.root_dir, "models")


class TenantManager:
    """
    Manages PSA tenants at ~/.psa/tenants/{tenant_id}/.

    Single-user deployments use the "default" tenant automatically.
    Multi-tenant deployments (e.g., server mode) create a tenant per user.
    """

    DEFAULT_TENANT_ID = "default"

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is not None:
            self._tenants_dir = Path(base_dir)
        else:
            self._tenants_dir = Path(os.path.expanduser("~/.psa/tenants"))

    def _tenant_root(self, tenant_id: str) -> Path:
        return self._tenants_dir / tenant_id

    def create(self, tenant_id: str, display_name: str = "", metadata: Optional[dict] = None) -> Tenant:
        """Create a new tenant directory structure. Idempotent."""
        _validate_tenant_id(tenant_id)
        root = self._tenant_root(tenant_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "atlas").mkdir(exist_ok=True)
        (root / "models").mkdir(exist_ok=True)

        tenant = Tenant(
            tenant_id=tenant_id,
            display_name=display_name or tenant_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            root_dir=str(root),
            metadata=metadata or {},
        )
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        """Return tenant if its directory exists, else None."""
        _validate_tenant_id(tenant_id)
        root = self._tenant_root(tenant_id)
        if not root.exists():
            return None
        return Tenant(
            tenant_id=tenant_id,
            display_name=tenant_id,
            created_at="",  # not persisted to disk, only used at runtime
            root_dir=str(root),
        )

    def get_or_create(self, tenant_id: str, display_name: str = "") -> Tenant:
        """Get existing tenant or create it if it doesn't exist."""
        existing = self.get(tenant_id)
        if existing is not None:
            return existing
        return self.create(tenant_id, display_name=display_name)

    def get_default(self) -> Tenant:
        """Get (or create) the default tenant."""
        return self.get_or_create(self.DEFAULT_TENANT_ID, display_name="Default")

    def list(self) -> List[Tenant]:
        """List all tenant directories found under the tenants root."""
        if not self._tenants_dir.exists():
            return []
        tenants = []
        for entry in sorted(self._tenants_dir.iterdir()):
            if entry.is_dir() and _TENANT_ID_RE.match(entry.name):
                tenants.append(
                    Tenant(
                        tenant_id=entry.name,
                        display_name=entry.name,
                        created_at="",
                        root_dir=str(entry),
                    )
                )
        return tenants

    def exists(self, tenant_id: str) -> bool:
        """Return True if the tenant directory exists."""
        try:
            _validate_tenant_id(tenant_id)
        except ValueError:
            return False
        return self._tenant_root(tenant_id).exists()
