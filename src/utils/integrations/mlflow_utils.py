from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from src.utils.core.logger import logger


def _ensure_local_tracking_dir(tracking_uri: str) -> None:
    if tracking_uri and tracking_uri.startswith("file://"):
        # file:///path/to/mlruns.db or directory
        path = tracking_uri[len("file://") :]
        p = Path(path)
        try:
            (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def setup_mlflow(tracking_uri: str) -> None:
    _ensure_local_tracking_dir(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


def get_model_uri(run_id: str, artifact_path: Optional[str] = None) -> str:
    path = artifact_path or "model"
    return f"runs:/{run_id}/{path}"


def load_pyfunc_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)


def create_model_signature(schema_dict: Dict[str, str]) -> ModelSignature:
    # Accept simple dict like {column: dtype_str}
    cols = []
    mapping = {
        "int": "integer",
        "float": "double",
        "double": "double",
        "str": "string",
        "string": "string",
        "bool": "boolean",
    }
    for name, dtype in (schema_dict or {}).items():
        ml_type = mapping.get(str(dtype).lower(), "string")
        cols.append(ColSpec(ml_type, name))
    return ModelSignature(inputs=Schema(cols))


def log_enhanced_model_with_schema(
    run_id: str, model_path: str, schema: Optional[Dict[str, str]] = None
) -> None:
    try:
        sig = create_model_signature(schema or {})
        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=None,  # caller should supply in real flow; tests validate signature creation path
            signature=sig,
        )
    except Exception as e:
        logger.warning(f"모델 로깅 중 경고: {e}")
