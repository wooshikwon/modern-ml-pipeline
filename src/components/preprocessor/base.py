"""
BasePreprocessor - 전처리기 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class BasePreprocessor(ABC):
    """
    DataFrame-First 아키텍처를 위한 전처리기 기본 클래스.

    핵심 원칙:
    1. 입출력은 항상 pd.DataFrame
    2. 컬럼명은 자체적으로 관리 (외부에서 추론하지 않음)
    3. 변환 후 컬럼명을 명시적으로 제공
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":
        """
        입력 데이터(X)로부터 변환에 필요한 파라미터들을 학습합니다.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 파라미터를 사용하여 데이터를 변환합니다.
        반드시 적절한 컬럼명을 가진 DataFrame을 반환해야 합니다.
        """
        pass

    def get_output_columns(self) -> List[str]:
        """
        전처리 후 출력 컬럼 목록 반환.
        fit() 호출 후에만 유효한 값을 반환합니다.

        기본 구현: _input_columns 또는 _fitted_columns 속성에서 반환.
        복잡한 변환이 있는 경우 하위 클래스에서 오버라이드.
        """
        if hasattr(self, "_input_columns") and self._input_columns:
            return list(self._input_columns)
        if hasattr(self, "_fitted_columns") and self._fitted_columns:
            return list(self._fitted_columns)
        return []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        fit과 transform을 순차적으로 수행합니다.
        """
        return self.fit(X, y).transform(X)

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """
        주어진 입력 컬럼명으로부터 변환 후 예상되는 출력 컬럼명을 반환합니다.

        기본 구현: 입력 컬럼명을 그대로 반환 (컬럼명이 보존되는 전처리기용)
        """
        return input_columns

    def preserves_column_names(self) -> bool:
        """
        이 전처리기가 원본 컬럼명을 보존하는지 여부를 반환합니다.

        기본값: True (대부분의 전처리기는 컬럼명을 보존함)
        """
        return True

    def get_application_type(self) -> str:
        """
        이 전처리기의 적용 유형을 반환합니다.

        - 'global': 모든 적합한 컬럼에 자동 적용 (missing value handling, scaler 등)
        - 'targeted': 특정 컬럼을 지정해서 적용 (encoder, feature engineering 등)

        기본값: 'targeted'
        """
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """
        global 타입 전처리기용: 주어진 DataFrame에서 이 전처리기가 적용 가능한 컬럼들을 반환합니다.

        기본 구현: 모든 컬럼 반환
        """
        return list(X.columns)
