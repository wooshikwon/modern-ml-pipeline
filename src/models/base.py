"""
BaseModel - ML 모델 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """
    ML 모델 클래스를 위한 기본 인터페이스입니다.
    이 라이브러리와 호환되는 모든 모델은 이 클래스를 상속받고,
    fit과 predict 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "BaseModel":
        """
        주어진 데이터로 모델을 학습시킵니다.

        Args:
            X (pd.DataFrame): 학습용 피처 데이터.
            y (pd.Series, optional): 학습용 타겟 데이터. Defaults to None.
            **kwargs: 모델별 추가 인자 (e.g., CausalML의 treatment).

        Returns:
            'BaseModel': 학습된 모델 인스턴스.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 모델을 사용하여 예측을 수행합니다.

        Args:
            X (pd.DataFrame): 예측할 피처 데이터.

        Returns:
            pd.DataFrame: 예측 결과.
        """
        pass
