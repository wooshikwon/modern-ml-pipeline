"""
BaseTrainer - 학습기 기본 인터페이스
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """
    모델 학습 및 평가 전체 과정을 관장하는 클래스의 추상 기본 클래스(ABC).
    """

    def __init__(self, settings: Any = None, **kwargs: Any):
        """
        Trainer 초기화

        Args:
            settings: 프로젝트 설정 객체 (선택사항, 다른 Base 클래스와 인터페이스 통일)
            **kwargs: 추가 키워드 인자
        """
        self.settings = settings

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
