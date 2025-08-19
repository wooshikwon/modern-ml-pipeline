from __future__ import annotations
from typing import Set

from src.settings import Settings


def _requires_pyarrow(settings: Settings) -> bool:
    try:
        src = settings.recipe.model.loader.source_uri.lower()
        return src.endswith(".parquet")
    except Exception:
        return False


def validate_dependencies(settings: Settings) -> None:
    """
    Validate runtime dependencies based on enabled features and recipe configuration.
    Missing packages must cause immediate ImportError.
    """
    required: Set[str] = set()

    # Adapter-based requirements
    try:
        adapter = settings.recipe.model.loader.adapter
    except Exception:
        adapter = None

    if adapter == "storage":
        if _requires_pyarrow(settings):
            required.add("pyarrow")
    elif adapter == "sql":
        required.add("sqlalchemy")

    # Feature store
    feature_store = getattr(settings, "feature_store", None)
    if feature_store and getattr(feature_store, "provider", None) == "feast":
        required.add("feast")

    # Hyperparameter tuning
    try:
        global_tuning = bool(getattr(settings.hyperparameter_tuning, "enabled", False))
    except Exception:
        global_tuning = False
    try:
        recipe_tuning = bool(getattr(settings.recipe.model.hyperparameter_tuning, "enabled", False))
    except Exception:
        recipe_tuning = False
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