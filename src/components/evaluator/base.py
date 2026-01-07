"""
BaseEvaluator - 평가기 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from src.settings import Settings


class BaseEvaluator(ABC):
    """
    모든 Evaluator가 따라야 할 표준 계약.
    각 Task Type별로 최적화된 평가 메트릭을 계산하는 Strategy Pattern의 기반.
    """

    def __init__(self, settings: "Settings"):
        """
        Evaluator 초기화

        Args:
            settings: Settings 객체 (전체 설정)
        """
        self.settings = settings
        self.data_interface = settings.recipe.data.data_interface
        self.task_choice = settings.recipe.task_choice

    @abstractmethod
    def evaluate(self, model, X, y, source_df=None) -> Dict[str, float]:
        """
        모델 평가 메트릭 계산

        Args:
            model: 학습된 모델 객체 (외부 라이브러리 모델)
            X: 테스트 피처 데이터
            y: 테스트 타겟 데이터 (clustering의 경우 None 가능)
            source_df: 원본 테스트 데이터프레임 (추가 컬럼 접근용, 선택사항)

        Returns:
            Dict[str, float]: 계산된 평가 메트릭들
        """
        pass
