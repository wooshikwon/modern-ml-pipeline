from __future__ import annotations
from typing import Dict, Type
from src.interface import BaseCalibrator
from src.utils.core.logger import logger


class CalibrationRegistry:
    """컴포넌트 레벨 Calibration 레지스트리 (엔진 의존성 제거)."""
    calibrators: Dict[str, Type[BaseCalibrator]] = {}

    @classmethod
    def register(cls, method_name: str, calibrator_class: Type[BaseCalibrator]):
        """
        Calibration 메서드 등록
        
        Args:
            method_name: Calibration 메서드 이름 (예: 'platt', 'isotonic')
            calibrator_class: BaseCalibrator를 상속받은 클래스
            
        Raises:
            TypeError: BaseCalibrator를 상속받지 않은 클래스인 경우
        """
        if not issubclass(calibrator_class, BaseCalibrator):
            raise TypeError(f"{calibrator_class.__name__} must be a subclass of BaseCalibrator")
        cls.calibrators[method_name] = calibrator_class
        logger.debug(f"[components] Calibrator registered: {method_name} -> {calibrator_class.__name__}")

    @classmethod
    def create(cls, method_name: str, **kwargs) -> BaseCalibrator:
        """
        Calibrator 인스턴스 생성
        
        Args:
            method_name: Calibration 메서드 이름
            **kwargs: Calibrator 생성자 파라미터
            
        Returns:
            BaseCalibrator: 생성된 calibrator 인스턴스
            
        Raises:
            ValueError: 등록되지 않은 메서드 이름인 경우
        """
        calibrator_class = cls.calibrators.get(method_name)
        if not calibrator_class:
            available = list(cls.calibrators.keys())
            raise ValueError(f"Unknown calibration method: '{method_name}'. Available methods: {available}")
        logger.debug(f"[components] Creating calibrator instance: {method_name}")
        return calibrator_class(**kwargs)

    @classmethod
    def get_available_methods(cls) -> list[str]:
        """등록된 모든 calibration 메서드 목록 반환."""
        return list(cls.calibrators.keys())

    @classmethod
    def get_calibrator_class(cls, method_name: str) -> Type[BaseCalibrator]:
        """Method name에 해당하는 Calibrator 클래스 반환."""
        calibrator_class = cls.calibrators.get(method_name)
        if not calibrator_class:
            available = list(cls.calibrators.keys())
            raise ValueError(f"Unknown calibration method: '{method_name}'. Available methods: {available}")
        return calibrator_class