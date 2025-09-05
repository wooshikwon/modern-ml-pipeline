"""
Adapter Registry - Self-registration pattern for data adapters.
각 어댑터 모듈에서 자동으로 자신을 등록하여 의존성을 줄입니다.
"""

from typing import Dict, Type, Any
from src.interface.base_adapter import BaseAdapter
from src.utils.system.logger import logger


class AdapterRegistry:
    """Data Adapter 등록 및 관리 클래스"""
    
    adapters: Dict[str, Type[BaseAdapter]] = {}
    
    @classmethod
    def register(cls, adapter_type: str, adapter_class: Type[BaseAdapter]):
        """어댑터를 레지스트리에 등록합니다.
        
        Args:
            adapter_type: 어댑터 타입 식별자 ('sql', 'storage', 'feature_store' 등)
            adapter_class: 어댑터 클래스
        """
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(f"어댑터 클래스는 BaseAdapter를 상속해야 합니다: {adapter_class}")
            
        cls.adapters[adapter_type] = adapter_class
        logger.debug(f"Adapter registered: {adapter_type} -> {adapter_class.__name__}")
    
    @classmethod
    def get_adapter(cls, adapter_type: str) -> Type[BaseAdapter]:
        """등록된 어댑터 클래스를 반환합니다.
        
        Args:
            adapter_type: 어댑터 타입 식별자
            
        Returns:
            어댑터 클래스
            
        Raises:
            KeyError: 등록되지 않은 어댑터 타입인 경우
        """
        if adapter_type not in cls.adapters:
            available = list(cls.adapters.keys())
            raise KeyError(f"Unknown adapter type: {adapter_type}. Available: {available}")
            
        return cls.adapters[adapter_type]
    
    @classmethod
    def list_adapters(cls) -> Dict[str, Type[BaseAdapter]]:
        """등록된 모든 어댑터를 반환합니다."""
        return cls.adapters.copy()
    
    @classmethod
    def create(cls, adapter_type: str, *args, **kwargs) -> BaseAdapter:
        """어댑터 인스턴스를 생성합니다.
        
        Args:
            adapter_type: 어댑터 타입 식별자
            *args, **kwargs: 어댑터 생성자에 전달할 인자
            
        Returns:
            어댑터 인스턴스
        """
        adapter_class = cls.get_adapter(adapter_type)
        return adapter_class(*args, **kwargs)