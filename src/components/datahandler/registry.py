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
    def get_handler_for_task(cls, task_choice: str, settings, model_class_path: str = None) -> BaseDataHandler:
        """
        Model catalog 기반 DataHandler 선택 (task_choice는 호환성 검증용으로만 사용)
        
        Args:
            task_choice: Recipe의 task_choice (검증용)
            settings: Settings 인스턴스  
            model_class_path: 모델 클래스 경로
            
        Returns:
            catalog 기반으로 선택된 DataHandler 인스턴스
        """
        # 🔍 모델 catalog에서 data_handler 정보 추출
        catalog_handler = cls._get_data_handler_from_catalog(model_class_path)
        
        if catalog_handler in cls.handlers:
            # 📋 Task와 Handler 호환성 검증 (선택사항)
            cls._validate_task_handler_compatibility(task_choice, catalog_handler)
            
            logger.info(f"🧠 Catalog 기반 핸들러 선택: {catalog_handler} (task: {task_choice})")
            # BaseDataHandler 인터페이스는 settings만 받도록 설계되어 있으므로 단일 인자 전달
            return cls.create(catalog_handler, settings)
        
        available = list(cls.handlers.keys())
        raise ValueError(f"지원하지 않는 data_handler: '{catalog_handler}'. 사용 가능한 핸들러: {available}")
    
    @classmethod 
    def _get_data_handler_from_catalog(cls, model_class_path: str) -> str:
        """
        모델 catalog에서 data_handler 추출
        
        Args:
            model_class_path: 모델 클래스 경로
            
        Returns:
            사용할 data_handler 이름
        """        
        if not model_class_path:
            return "tabular"  # 기본값
            
        catalog = cls._load_model_catalog(model_class_path)
        if catalog and 'data_handler' in catalog:
            handler = catalog['data_handler']
            logger.debug(f"📋 Catalog에서 data_handler 발견: {handler}")
            return handler
        
        # Fallback: 기본값
        logger.debug(f"📋 Catalog에 data_handler가 없어 기본값 사용: tabular")
        return "tabular"
    
    @classmethod
    def _load_model_catalog(cls, model_class_path: str) -> dict:
        """
        모델 클래스 경로에서 catalog 정보 로드
        
        Args:
            model_class_path: 모델 클래스 경로
            
        Returns:
            catalog 딕셔너리 (로드 실패시 빈 딕셔너리)
        """
        if not model_class_path:
            return {}
            
        try:
            import yaml
            from pathlib import Path
            
            # 클래스 경로에서 catalog 파일 경로 추론
            # 예: "src.models.custom.lstm_timeseries.LSTMTimeSeries" → "DeepLearning/LSTMTimeSeries.yaml"
            parts = model_class_path.split('.')
            if len(parts) >= 2:
                class_name = parts[-1]  # LSTMTimeSeries
                
                # Catalog 디렉토리에서 해당 파일 찾기
                catalog_root = Path(__file__).parent.parent.parent / "models" / "catalog"
                
                # 모든 task 디렉토리에서 검색
                for task_dir in catalog_root.iterdir():
                    if task_dir.is_dir():
                        catalog_file = task_dir / f"{class_name}.yaml"
                        if catalog_file.exists():
                            with open(catalog_file, 'r', encoding='utf-8') as f:
                                return yaml.safe_load(f) or {}
                            
                logger.debug(f"📋 Catalog 파일을 찾을 수 없음: {class_name}")
                
        except Exception as e:
            logger.warning(f"⚠️  Catalog 로드 실패: {model_class_path}, Error: {e}")
            
        return {}
    
    @classmethod
    def _validate_task_handler_compatibility(cls, task_choice: str, handler_type: str):
        """Task와 Handler 호환성 검증 (선택사항)"""
        # 예: timeseries task인데 tabular handler 사용 시 경고
        if task_choice == "timeseries" and handler_type == "tabular":
            logger.warning("⚠️ Timeseries task에 tabular handler 사용. 의도한 것이 맞나요?")
        elif task_choice in ["classification", "regression", "clustering", "causal"] and handler_type == "deeplearning":
            logger.info("🧠 딥러닝 모델을 사용한 {}. deeplearning handler를 사용합니다.".format(task_choice))
    
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