"""
Factory Registry Pattern Implementation
Blueprint v17.0 - Architecture Excellence

이 모듈은 Blueprint 원칙 3 "URI 기반 동작 및 동적 팩토리"의 완전한 구현을 위한
확장성 있는 Registry 패턴을 제공합니다.

핵심 특징:
- 데코레이터 기반 자동 어댑터 등록
- if-else 분기 없는 완전한 확장성
- 하위 호환성 보장
- 명확한 에러 메시지
"""

import importlib
from typing import Dict, Type, Optional
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger


class AdapterRegistry:
    """
    완전히 확장적인 어댑터 등록 시스템
    Blueprint 원칙 3: "URI 기반 동작 및 동적 팩토리"의 핵심 구현
    """
    
    _adapters: Dict[str, Type[BaseAdapter]] = {}
    
    @classmethod
    def register(cls, adapter_type: str):
        """
        어댑터 등록 데코레이터
        
        Args:
            adapter_type: 어댑터 타입 (e.g., "postgresql", "redis", "bigquery")
            
        Returns:
            데코레이터 함수
            
        Example:
            @AdapterRegistry.register("postgresql")
            class PostgreSQLAdapter(BaseAdapter):
                pass
        """
        def decorator(adapter_class: Type[BaseAdapter]):
            cls._adapters[adapter_type] = adapter_class
            logger.info(f"어댑터 등록됨: {adapter_type} -> {adapter_class.__name__}")
            return adapter_class
        return decorator
    
    @classmethod
    def create(cls, adapter_type: str, settings: Settings, **kwargs) -> BaseAdapter:
        """
        동적 어댑터 생성
        
        Args:
            adapter_type: 어댑터 타입
            settings: 설정 객체
            **kwargs: 어댑터 생성에 필요한 추가 인자
            
        Returns:
            BaseAdapter: 생성된 어댑터 인스턴스
            
        Raises:
            ValueError: 등록되지 않은 어댑터 타입인 경우
        """
        if adapter_type not in cls._adapters:
            available_types = list(cls._adapters.keys())
            raise ValueError(
                f"등록되지 않은 어댑터 타입: '{adapter_type}'\n"
                f"사용 가능한 타입: {available_types}"
            )
        
        adapter_class = cls._adapters[adapter_type]
        logger.info(f"어댑터 생성: {adapter_type} -> {adapter_class.__name__}")
        
        try:
            # 어댑터 인스턴스 생성 시도
            return adapter_class(settings, **kwargs)
        except Exception as e:
            logger.error(f"어댑터 생성 실패: {adapter_type}, 오류: {e}")
            raise ValueError(f"어댑터 생성 실패: {adapter_type}") from e
    
    @classmethod
    def get_registered_adapters(cls) -> Dict[str, Type[BaseAdapter]]:
        """등록된 모든 어댑터 목록 반환"""
        return cls._adapters.copy()
    
    @classmethod
    def is_registered(cls, adapter_type: str) -> bool:
        """어댑터 타입이 등록되어 있는지 확인"""
        return adapter_type in cls._adapters


# =============================================================================
# 자동 어댑터 등록 시스템 (기존 import 매핑 기반)
# =============================================================================

def auto_register_adapters():
    """
    기존 어댑터들을 자동으로 등록하는 함수
    하위 호환성 보장을 위해 기존 import 매핑을 활용
    """
    # 기존 import 매핑 (factory.py에서 이동)
    adapter_import_mapping = {
        "FileSystemAdapter": "src.utils.adapters.file_system_adapter",
        "BigQueryAdapter": "src.utils.adapters.bigquery_adapter", 
        "GCSAdapter": "src.utils.adapters.gcs_adapter",
        "S3Adapter": "src.utils.adapters.s3_adapter",
        "PostgreSQLAdapter": "src.utils.adapters.postgresql_adapter",
        "RedisAdapter": "src.utils.adapters.redis_adapter",
        "FeatureStoreAdapter": "src.utils.adapters.feature_store_adapter",
        "OptunaAdapter": "src.utils.adapters.optuna_adapter",
    }
    
    # 어댑터 타입 매핑 (클래스명 -> 어댑터 타입)
    adapter_type_mapping = {
        "FileSystemAdapter": "filesystem",
        "BigQueryAdapter": "bigquery",
        "GCSAdapter": "gcs",
        "S3Adapter": "s3",
        "PostgreSQLAdapter": "postgresql",
        "RedisAdapter": "redis",
        "FeatureStoreAdapter": "feature_store",
        "OptunaAdapter": "optuna",
    }
    
    for class_name, module_path in adapter_import_mapping.items():
        try:
            # 동적 모듈 import
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            
            # 어댑터 타입 결정
            adapter_type = adapter_type_mapping.get(class_name, class_name.lower())
            
            # Registry에 등록
            AdapterRegistry._adapters[adapter_type] = adapter_class
            logger.info(f"자동 등록됨: {adapter_type} -> {class_name}")
            
        except ImportError as e:
            logger.warning(f"어댑터 자동 등록 실패 (정상적인 상황): {class_name}, 오류: {e}")
        except Exception as e:
            logger.error(f"어댑터 자동 등록 중 예상치 못한 오류: {class_name}, 오류: {e}")


# 모듈 로드 시 자동 등록 실행
auto_register_adapters() 