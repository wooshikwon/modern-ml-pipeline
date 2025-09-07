"""
DataHandler Registry - 데이터 핸들러 중앙 관리

Registry 패턴을 통해 task_type별로 적절한 DataHandler를 자동으로 매핑하고 생성합니다.
"""

from typing import Dict, Type, Optional
from src.interface import BaseDataHandler
from src.utils.system.logger import logger


class DataHandlerRegistry:
    """DataHandler 중앙 레지스트리"""
    handlers: Dict[str, Type[BaseDataHandler]] = {}
    
    @classmethod
    def register(cls, handler_type: str, handler_class: Type[BaseDataHandler]):
        """
        DataHandler 등록
        
        Args:
            handler_type: 핸들러 타입 (tabular, timeseries, deeplearning 등)
            handler_class: BaseDataHandler를 상속받은 클래스
        """
        if not issubclass(handler_class, BaseDataHandler):
            raise TypeError(f"{handler_class.__name__} must be a subclass of BaseDataHandler")
        cls.handlers[handler_type] = handler_class
        logger.debug(f"DataHandler registered: {handler_type} -> {handler_class.__name__}")
    
    @classmethod
    def create(cls, handler_type: str, *args, **kwargs) -> BaseDataHandler:
        """
        DataHandler 인스턴스 생성
        
        Args:
            handler_type: 핸들러 타입
            *args, **kwargs: 핸들러 생성자 인자들
            
        Returns:
            DataHandler 인스턴스
        """
        handler_class = cls.handlers.get(handler_type)
        if not handler_class:
            available = list(cls.handlers.keys())
            raise ValueError(f"Unknown handler type: '{handler_type}'. Available: {available}")
        return handler_class(*args, **kwargs)
    
    @classmethod
    def get_handler_for_task(cls, task_type: str, settings) -> BaseDataHandler:
        """
        task_type에 따른 자동 handler 매핑
        
        Args:
            task_type: ML 태스크 타입
            settings: Settings 인스턴스
            
        Returns:
            해당 task_type에 적합한 DataHandler 인스턴스
        """
        # task_type -> handler_type 매핑
        handler_mapping = {
            "classification": "tabular",
            "regression": "tabular",
            "clustering": "tabular", 
            "causal": "tabular",
            "timeseries": "timeseries"
            # 향후 "deeplearning": "deeplearning" 추가
        }
        
        handler_type = handler_mapping.get(task_type, "tabular")
        
        try:
            return cls.create(handler_type, settings)
        except ValueError as e:
            # Fallback to tabular handler
            logger.warning(f"Handler for task '{task_type}' not found, falling back to tabular handler")
            return cls.create("tabular", settings)
    
    @classmethod
    def get_available_handlers(cls) -> Dict[str, str]:
        """
        등록된 핸들러 목록 반환
        
        Returns:
            {handler_type: handler_class_name} 딕셔너리
        """
        return {handler_type: handler_class.__name__ for handler_type, handler_class in cls.handlers.items()}
    
    @classmethod
    def clear(cls):
        """테스트용: 등록된 핸들러 모두 제거"""
        cls.handlers.clear()