# mmp/utils/integrations/mlflow_integration.py
"""MLflow integration re-export 파사드.

기존 ``from mmp.utils.integrations.mlflow_integration import setup_mlflow`` 및
``from mmp.utils.integrations import mlflow_integration as mlflow_utils`` 패턴을
100% 하위 호환으로 유지하면서, 실제 구현은 아래 두 모듈에 분리되어 있다:

- ``mlflow_tracker.py``  — ``MLflowTracker`` (실험 lifecycle, 로깅, 아티팩트)
- ``mlflow_signature.py`` — ``MLflowSignatureBuilder`` (시그니처, 스키마 메타데이터)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, List, Optional

import pandas as pd
from mlflow.models.signature import ModelSignature

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.pyfunc import PyFuncModel

    from mmp.settings import Settings

# ---------------------------------------------------------------------------
# Re-export classes
# ---------------------------------------------------------------------------

from mmp.utils.integrations.mlflow_signature import (  # noqa: E402
    MLflowSignatureBuilder,
    _infer_pandas_dtype_to_mlflow_type,
)
from mmp.utils.integrations.mlflow_tracker import MLflowTracker  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level singletons (for backward-compatible function API)
# ---------------------------------------------------------------------------

_default_tracker = MLflowTracker()
_default_signature_builder = MLflowSignatureBuilder()


# ---------------------------------------------------------------------------
# Backward-compatible module-level functions
# ---------------------------------------------------------------------------
# Every existing ``from mmp.utils.integrations.mlflow_integration import X``
# and ``mlflow_utils.X(...)`` call continues to work unchanged.


def generate_unique_run_name(base_run_name: str) -> str:
    """See :meth:`MLflowTracker.generate_unique_run_name`."""
    return MLflowTracker.generate_unique_run_name(base_run_name)


def setup_mlflow(settings: Settings) -> None:
    """See :meth:`MLflowTracker.setup_mlflow`."""
    _default_tracker.setup_mlflow(settings)


@contextmanager
def start_run(settings: Settings, run_name: str = "run") -> Generator[Run, None, None]:
    """See :meth:`MLflowTracker.start_run`."""
    with _default_tracker.start_run(settings, run_name=run_name) as run:
        yield run


def get_latest_run_id(settings: Settings, experiment_name: str) -> str:
    """See :meth:`MLflowTracker.get_latest_run_id`."""
    return _default_tracker.get_latest_run_id(settings, experiment_name)


def get_model_uri(run_id: str, artifact_path: str = "model") -> str:
    """See :meth:`MLflowTracker.get_model_uri`."""
    return MLflowTracker.get_model_uri(run_id, artifact_path)


def load_pyfunc_model(settings: Settings, model_uri: str) -> PyFuncModel:
    """See :meth:`MLflowTracker.load_pyfunc_model`."""
    return _default_tracker.load_pyfunc_model(settings, model_uri)


def download_artifacts(
    settings: Settings, run_id: str, artifact_path: str, dst_path: str | None = None
) -> str:
    """See :meth:`MLflowTracker.download_artifacts`."""
    return _default_tracker.download_artifacts(settings, run_id, artifact_path, dst_path)


def create_model_signature(
    input_df: pd.DataFrame, output_df: pd.DataFrame, params: dict | None = None
) -> ModelSignature:
    """See :meth:`MLflowSignatureBuilder.create_model_signature`."""
    return MLflowSignatureBuilder.create_model_signature(input_df, output_df, params)


def create_enhanced_model_signature_with_schema(
    training_df: pd.DataFrame, data_interface_config: dict
) -> tuple[ModelSignature, dict]:
    """See :meth:`MLflowSignatureBuilder.create_signature_with_schema`."""
    return MLflowSignatureBuilder.create_signature_with_schema(training_df, data_interface_config)


def log_enhanced_model_with_schema(
    python_model,
    signature: ModelSignature,
    data_schema: dict,
    input_example: pd.DataFrame,
    pip_requirements: Optional[List[str]] = None,
) -> None:
    """See :meth:`MLflowSignatureBuilder.log_enhanced_model_with_schema`."""
    MLflowSignatureBuilder.log_enhanced_model_with_schema(
        python_model, signature, data_schema, input_example, pip_requirements
    )


def log_training_results(
    settings: Settings, metrics: dict, training_results: dict
) -> None:
    """See :meth:`MLflowTracker.log_training_results`."""
    _default_tracker.log_training_results(settings, metrics, training_results)


# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__ = [
    # Classes
    "MLflowTracker",
    "MLflowSignatureBuilder",
    # Module-level functions (backward compat)
    "generate_unique_run_name",
    "setup_mlflow",
    "start_run",
    "get_latest_run_id",
    "get_model_uri",
    "load_pyfunc_model",
    "download_artifacts",
    "create_model_signature",
    "create_enhanced_model_signature_with_schema",
    "log_enhanced_model_with_schema",
    "log_training_results",
    # Private but used in tests
    "_infer_pandas_dtype_to_mlflow_type",
]
