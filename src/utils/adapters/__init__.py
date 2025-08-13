"""Backward-compat imports for adapters (moved to src/components/_adapters/plugins)."""

from src.components._adapters.plugins.sql_adapter import SqlAdapter  # noqa: F401
from src.components._adapters.plugins.storage_adapter import StorageAdapter  # noqa: F401
try:
    from src.components._adapters.plugins.feast_adapter import FeastAdapter  # noqa: F401
except Exception:
    FeastAdapter = None

__all__ = ["SqlAdapter", "StorageAdapter", "FeastAdapter"]