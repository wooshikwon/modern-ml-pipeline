from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from src.components.preprocessor import Preprocessor
# BaseModel import 제거: 외부 라이브러리 직접 사용으로 전환


class BaseTrainer(ABC):
    """
    모델 학습 및 평가 전체 과정을 관장하는 클래스의 추상 기본 클래스(ABC).
    """

    @abstractmethod
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        model: Any,
        additional_data: dict | None = None,
    ) -> tuple[Any, dict]:
        """순수 학습만 수행하여 (학습된 모델, 학습 메타데이터) 튜플을 반환합니다."""
        raise NotImplementedError
