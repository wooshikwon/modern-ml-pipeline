"""
외부 시스템 연동을 위한 통합 어댑터 모음

이 패키지는 `SqlAdapter`, `StorageAdapter`, `FeastAdapter` 등
현대화된 통합 어댑터를 포함합니다.
"""

from .sql_adapter import SqlAdapter  # noqa: F401
from .storage_adapter import StorageAdapter  # noqa: F401
try:
    from .feast_adapter import FeastAdapter  # noqa: F401
except Exception:
    FeastAdapter = None  # 선택 의존성 미설치 시 무시

__all__ = [
    "SqlAdapter",
    "StorageAdapter",
    "FeastAdapter",
] 