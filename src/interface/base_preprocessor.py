from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BasePreprocessor(ABC):
    """
    피처 변환을 위한 추상 기본 클래스(ABC).

    모든 Preprocessor 구현체는 이 클래스를 상속받아 `fit`과 `transform` 메서드를 구현해야 합니다.
    이는 scikit-learn의 Preprocessor API와 유사한 인터페이스를 제공하여 일관성을 유지합니다.
    """
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BasePreprocessor':
        """
        입력 데이터(X)로부터 변환에 필요한 파라미터들을 학습합니다.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 파라미터를 사용하여 데이터를 변환합니다.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        fit과 transform을 순차적으로 수행합니다.
        """
        return self.fit(X, y).transform(X)
