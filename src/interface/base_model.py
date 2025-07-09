from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Uplift 모델의 기본 인터페이스(추상 클래스)입니다."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        주어진 데이터로 모델을 학습시키는 추상 메서드입니다.

        Args:
            X: 피처 데이터프레임
            y: 타겟(결과) 시리즈
            treatment: 처치(실험군/대조군) 그룹 시리즈
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        새로운 데이터에 대한 Uplift 점수를 예측하는 추상 메서드입니다.
        """
        raise NotImplementedError
