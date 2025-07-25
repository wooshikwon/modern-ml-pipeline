"""
외부 시스템 연동을 위한 데이터 어댑터들

이 모듈은 다양한 외부 시스템(BigQuery, GCS, S3, Redis 등)과의 
데이터 읽기/쓰기를 담당하는 어댑터들을 포함합니다.
"""

from .bigquery_adapter import BigQueryAdapter
from .gcs_adapter import GCSAdapter
from .s3_adapter import S3Adapter
from .file_system_adapter import FileSystemAdapter
from .redis_adapter import RedisAdapter

__all__ = [
    'BigQueryAdapter',
    'GCSAdapter', 
    'S3Adapter',
    'FileSystemAdapter',
    'RedisAdapter',
    'HAS_REDIS'
] 