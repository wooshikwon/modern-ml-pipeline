"""
DataHandler Registry - 데이터 핸들러 전용 Registry.
"""

from typing import Dict, Type

from src.components.base_registry import BaseRegistry
from src.utils.core.logger import logger

from .base import BaseDataHandler
from .catalog_utils import get_data_handler_type_from_catalog, validate_task_handler_compatibility


class DataHandlerRegistry(BaseRegistry[BaseDataHandler]):
    """DataHandler 전용 Registry."""

    _registry: Dict[str, Type[BaseDataHandler]] = {}
    _base_class = BaseDataHandler

    @classmethod
    def get_handler_for_task(
        cls, task_choice: str, settings, model_class_path: str = None
    ) -> BaseDataHandler:
        """
        Model catalog 기반 DataHandler 선택.

        Args:
            task_choice: Recipe의 task_choice (호환성 검증용)
            settings: Settings 인스턴스
            model_class_path: 모델 클래스 경로

        Returns:
            catalog 기반으로 선택된 DataHandler 인스턴스
        """
        catalog_handler = get_data_handler_type_from_catalog(model_class_path)

        # 핸들러별 변환 방식 설명
        handler_descriptions = {
            "tabular": "2D 테이블 유지",
            "timeseries": "2D 유지 + lag/rolling 피처 자동 생성",
            "sequence": "2D→3D sliding window 변환",
        }

        if catalog_handler in cls._registry:
            validate_task_handler_compatibility(task_choice, catalog_handler)
            desc = handler_descriptions.get(catalog_handler, "")
            logger.info(
                f"[DATA] 핸들러 선택: {catalog_handler} ({desc}) - task: {task_choice}"
            )
            return cls.create(catalog_handler, settings)

        available = cls.list_keys()
        raise ValueError(
            f"지원하지 않는 data_handler: '{catalog_handler}'. 사용 가능한 핸들러: {available}"
        )
