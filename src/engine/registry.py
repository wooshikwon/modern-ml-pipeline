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
from typing import Dict, Type, Optional

from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger


class AdapterRegistry:
    """
    단순하고 명시적인 어댑터 등록 및 생성 시스템.
    """
    _adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: Type[BaseAdapter]):
        """
        어댑터를 Registry에 명시적으로 등록합니다.
        
        Args:
            adapter_type: 어댑터 타입 (e.g., "sql", "storage", "feature_store")
            adapter_class: 등록할 어댑터 클래스
        """
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(f"{adapter_class.__name__} must be a subclass of BaseAdapter")
        
        cls._adapters[adapter_type] = adapter_class
        logger.info(f"어댑터 등록됨: {adapter_type} -> {adapter_class.__name__}")

    @classmethod
    def create(cls, adapter_type: str, settings: Settings, **kwargs) -> BaseAdapter:
        """
        등록된 어댑터의 인스턴스를 생성합니다.
        
        Args:
            adapter_type: 생성할 어댑터 타입
            settings: 설정 객체
            **kwargs: 어댑터 생성에 필요한 추가 인자
            
        Returns:
            생성된 어댑터 인스턴스
            
        Raises:
            ValueError: 등록되지 않은 어댑터 타입인 경우
        """
        adapter_class = cls._adapters.get(adapter_type)
        if not adapter_class:
            available_types = list(cls._adapters.keys())
            raise ValueError(
                f"등록되지 않은 어댑터 타입: '{adapter_type}'\n"
                f"사용 가능한 타입: {available_types}"
            )
        
        logger.info(f"어댑터 생성: {adapter_type} -> {adapter_class.__name__}")
        
        try:
            return adapter_class(settings=settings, **kwargs)
        except Exception as e:
            logger.error(f"어댑터 '{adapter_type}' 생성 실패: {e}", exc_info=True)
            raise ValueError(f"어댑터 '{adapter_type}' 생성에 실패했습니다.") from e

    @classmethod
    def get_registered_adapters(cls) -> Dict[str, Type[BaseAdapter]]:
        """등록된 모든 어댑터의 복사본을 반환합니다."""
        return cls._adapters.copy()

    @classmethod
    def is_registered(cls, adapter_type: str) -> bool:
        """어댑터 타입이 등록되어 있는지 확인합니다."""
        return adapter_type in cls._adapters


def register_all_adapters():
    """
    시스템에서 사용되는 모든 어댑터를 명시적으로 등록합니다.
    선택적 의존성은 try-except 구문을 통해 우아하게 처리합니다.
    """
    logger.info("모든 코어 어댑터 등록을 시작합니다...")

    # Phase R4에서 구현될 통합 어댑터 (구조만 미리 정의)
    # try:
    #     from src.utils.adapters.sql_adapter import SqlAdapter
    #     AdapterRegistry.register("sql", SqlAdapter)
    # except ImportError:
    #     logger.warning("SqlAdapter를 찾을 수 없어 등록을 건너뜁니다. (선택적 의존성)")

    # try:
    #     from src.utils.adapters.storage_adapter import StorageAdapter
    #     AdapterRegistry.register("storage", StorageAdapter)
    # except ImportError:
    #     logger.warning("StorageAdapter를 찾을 수 없어 등록을 건너뜁니다. (선택적 의존성)")

    # Phase R3에서 구현될 Feast 어댑터
    # try:
    #     from src.utils.adapters.feast_adapter import FeastAdapter
    #     AdapterRegistry.register("feature_store", FeastAdapter)
    # except ImportError:
    #     logger.warning("FeastAdapter를 찾을 수 없어 등록을 건너뜁니다. (선택적 의존성)")
    
    # 임시: 리팩토링 과정에서 기존 어댑터들을 임시로 등록
    # 최종적으로는 통합 어댑터로 대체된 후 이 부분은 제거됩니다.
    _register_legacy_adapters_temporarily()


def _register_legacy_adapters_temporarily():
    """리팩토링 과도기 동안 기존 어댑터들을 임시로 등록하는 함수."""
    legacy_adapters = {
        "filesystem": "src.utils.adapters.file_system_adapter.FileSystemAdapter",
        "bigquery": "src.utils.adapters.bigquery_adapter.BigQueryAdapter",
        "gcs": "src.utils.adapters.gcs_adapter.GCSAdapter",
        "s3": "src.utils.adapters.s3_adapter.S3Adapter",
        "postgresql": "src.utils.adapters.postgresql_adapter.PostgreSQLAdapter",
        "redis": "src.utils.adapters.redis_adapter.RedisAdapter",
        "feature_store": "src.utils.adapters.feature_store_adapter.FeatureStoreAdapter",
    }
    
    import importlib
    
    for adapter_type, class_path in legacy_adapters.items():
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            AdapterRegistry.register(adapter_type, adapter_class)
        except ImportError:
            logger.debug(f"레거시 어댑터 '{adapter_type}'를 찾을 수 없어 등록을 건너뜁니다.")
        except Exception as e:
            logger.error(f"레거시 어댑터 '{adapter_type}' 등록 중 예상치 못한 오류 발생: {e}")


# 모듈 로드 시 모든 어댑터를 등록합니다.
register_all_adapters() 