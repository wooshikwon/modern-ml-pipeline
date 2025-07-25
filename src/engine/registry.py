"""
Factory Registry Pattern Implementation
Blueprint - Architecture Excellence

이 모듈은 Blueprint 원칙 3 "단순성과 명시성의 원칙"의 완전한 구현을 위한
확장성 있는 Registry 패턴을 제공합니다.

핵심 특징:
- 복잡성 제거: 데코레이터, 메타클래스, 자동 스캔 등 암시적 로직 배제
- 명시적 등록: 필요한 어댑터를 한 곳에서 명시적으로 등록
- 단순성: 순수한 딕셔너리 기반으로 동작하여 예측 가능성 극대화
- 확장성: 새로운 어댑터 추가 시, Factory 수정 없이 이 파일에 한 줄만 추가하면 됨
"""
from __future__ import annotations
import importlib
from typing import Dict, Type, TYPE_CHECKING
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.interface.base_adapter import BaseAdapter
    from src.settings import Settings


class AdapterRegistry:
    """
    단순하고 명시적인 어댑터 등록 및 생성 시스템.
    Blueprint 원칙 3: "단순성과 명시성의 원칙"을 구현합니다.
    """
    _adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: Type[BaseAdapter]):
        """
        어댑터를 Registry에 명시적으로 등록합니다.
        데코레이터나 자동 스캔이 아닌, 직접 호출 방식입니다.
        """
        from src.interface.base_adapter import BaseAdapter
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(f"{adapter_class.__name__} must be a subclass of BaseAdapter")
        
        cls._adapters[adapter_type] = adapter_class
        logger.debug(f"Adapter registered: {adapter_type} -> {adapter_class.__name__}")

    @classmethod
    def create(cls, adapter_type: str, settings: Settings, **kwargs) -> BaseAdapter:
        """등록된 어댑터의 인스턴스를 생성합니다."""
        adapter_class = cls._adapters.get(adapter_type)
        if not adapter_class:
            available = list(cls._adapters.keys())
            raise ValueError(f"Unknown adapter type: '{adapter_type}'. Available types: {available}")
        
        logger.debug(f"Creating adapter instance: {adapter_type}")
        return adapter_class(settings=settings, **kwargs)


def _register_legacy_adapters_temporarily():
    """
    Phase 2에서 통합 어댑터가 구현되기 전까지, 기존 개별 어댑터들을 임시로 등록합니다.
    순환 참조 방지를 위해 importlib를 사용합니다.
    """
    legacy_adapters = {
        "filesystem": "src.utils.adapters.file_system_adapter.FileSystemAdapter",
        "bigquery": "src.utils.adapters.bigquery_adapter.BigQueryAdapter",
        "gcs": "src.utils.adapters.gcs_adapter.GCSAdapter",
        "s3": "src.utils.adapters.s3_adapter.S3Adapter",
        "postgresql": "src.utils.adapters.postgresql_adapter.PostgreSQLAdapter",
        "redis": "src.utils.adapters.redis_adapter.RedisAdapter",
        "feature_store": "src.utils.adapters.feature_store_adapter.FeatureStoreAdapter",
    }
    
    for adapter_type, class_path in legacy_adapters.items():
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            AdapterRegistry.register(adapter_type, adapter_class)
        except ImportError:
            logger.debug(f"Legacy adapter '{adapter_type}' not found, skipping registration.")
        except Exception as e:
            logger.error(f"Error registering legacy adapter '{adapter_type}': {e}")


def register_all_adapters():
    """시스템에서 사용하는 모든 어댑터를 명시적으로 등록합니다."""
    logger.info("Registering all core adapters...")
    try:
        from src.utils.adapters.sql_adapter import SqlAdapter
        AdapterRegistry.register("sql", SqlAdapter)
    except ImportError:
        logger.warning("SqlAdapter could not be imported. Skipping registration.")

    try:
        from src.utils.adapters.storage_adapter import StorageAdapter
        AdapterRegistry.register("storage", StorageAdapter)
    except ImportError:
        logger.warning("StorageAdapter could not be imported. Skipping registration.")

    try:
        from src.utils.adapters.feast_adapter import FeastAdapter
        AdapterRegistry.register("feature_store", FeastAdapter)
    except ImportError:
        logger.warning("FeastAdapter could not be imported. Skipping registration.")
    
    logger.info("Core adapter registration complete.")

# 모듈 로드 시 모든 어댑터를 등록합니다.
register_all_adapters() 