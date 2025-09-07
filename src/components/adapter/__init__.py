"""
Data Adapters Package - Registry pattern with self-registration.

이 패키지는 다양한 데이터 소스와의 연동을 위한 어댑터들을 제공합니다.
각 어댑터는 임포트 시점에 자동으로 레지스트리에 등록됩니다.
"""

from .registry import AdapterRegistry

# 어댑터 모듈들을 임포트하여 자동 등록을 트리거합니다
from .modules.storage_adapter import StorageAdapter
from .modules.sql_adapter import SqlAdapter

try:
    from .modules.feast_adapter import FeastAdapter
except ImportError:
    # Feast는 선택적 의존성이므로 임포트 실패 시 무시
    FeastAdapter = None

# BigQuery 어댑터는 선택 의존성
try:
    from .modules.bigquery_adapter import BigQueryAdapter
except Exception:
    BigQueryAdapter = None

# 공개 API
__all__ = [
    "AdapterRegistry",
    "StorageAdapter", 
    "SqlAdapter",
    "FeastAdapter",
    "BigQueryAdapter",
]