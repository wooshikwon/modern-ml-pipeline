"""Serving Module Public API

FastAPI 의존성은 lazy import로 처리.
mmp serve-api 실행 시에만 실제 import가 발생한다.
"""


def run_api_server(*args, **kwargs):
    """API 서버 실행. FastAPI/uvicorn은 이 시점에 import된다."""
    try:
        from .router import run_api_server as _run
    except ImportError:
        raise ImportError(
            "API 서빙에 필요한 패키지가 설치되지 않았습니다.\n"
            "설치: pip install 'modern-ml-pipeline[serving]'"
        )
    return _run(*args, **kwargs)


def __getattr__(name):
    """Lazy import for serving schemas."""
    _schema_names = {
        "BatchPredictionResponse",
        "HealthCheckResponse",
        "ModelMetadataResponse",
        "OptimizationHistoryResponse",
        "ReadyCheckResponse",
    }
    if name in _schema_names:
        try:
            from . import schemas
            return getattr(schemas, name)
        except ImportError:
            raise ImportError(
                f"'{name}' 사용에 필요한 패키지가 설치되지 않았습니다.\n"
                "설치: pip install 'modern-ml-pipeline[serving]'"
            )
    raise AttributeError(f"module 'mmp.serving' has no attribute {name!r}")


__all__ = [
    "run_api_server",
    "BatchPredictionResponse",
    "HealthCheckResponse",
    "ReadyCheckResponse",
    "ModelMetadataResponse",
    "OptimizationHistoryResponse",
]
