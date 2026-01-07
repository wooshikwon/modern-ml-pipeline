from __future__ import annotations

import builtins as _builtins
from typing import Set

from src.settings import Settings

# Allow tests to patch this module-local symbol.
__import__ = _builtins.__import__


def _requires_pyarrow(settings: Settings) -> bool:
    """Return True if the configured data source URI points to a parquet file.

    Schema-aligned access path: settings.recipe.data.loader.source_uri
    """
    try:
        src = settings.recipe.data.loader.source_uri
        if not isinstance(src, str):
            return False
        return src.lower().endswith(".parquet")
    except Exception:
        return False


def validate_dependencies(settings: Settings) -> None:
    """
    Validate runtime dependencies based on enabled features and recipe configuration.
    Missing packages must cause immediate ImportError.
    """
    required: Set[str] = set()

    # Adapter-based requirements (respect Config schema)
    try:
        adapter_type = getattr(settings.config.data_source, "adapter_type", None)
    except Exception:
        adapter_type = None

    if adapter_type == "sql":
        required.add("sqlalchemy")

    # Parquet requirement applies regardless of adapter
    if _requires_pyarrow(settings):
        required.add("pyarrow")

    # Feature store
    feature_store = getattr(settings, "feature_store", None)
    if feature_store and getattr(feature_store, "provider", None) == "feast":
        required.add("feast")

    # Hyperparameter tuning
    try:
        global_tuning = bool(
            getattr(getattr(settings, "hyperparameter_tuning", None), "enabled", False)
        )
    except Exception:
        global_tuning = False

    # Support both legacy (model.hyperparameter_tuning.enabled) and current (model.hyperparameters.tuning_enabled)
    recipe_tuning = False
    try:
        recipe_tuning = bool(
            getattr(getattr(settings.recipe.model, "hyperparameter_tuning", None), "enabled", False)
        )
    except Exception:
        pass
    try:
        # New schema
        recipe_tuning = recipe_tuning or bool(
            getattr(
                getattr(settings.recipe.model, "hyperparameters", None), "tuning_enabled", False
            )
        )
    except Exception:
        pass
    # Require optuna only when BOTH global and recipe toggles are enabled
    if global_tuning and recipe_tuning:
        required.add("optuna")

    # Serving
    try:
        if getattr(settings.serving, "enabled", False):
            required.update({"fastapi", "uvicorn"})
    except Exception:
        pass

    # Enforce
    missing = []
    for pkg in sorted(required):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(f"필수 패키지가 설치되지 않았습니다: {missing}")
