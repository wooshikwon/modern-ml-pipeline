# src/components/_preprocessor/_steps/_imputer.py
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import pandas as pd
import numpy as np
from src.interface import BasePreprocessor
from ..registry import PreprocessorStepRegistry

class SimpleImputerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 SimpleImputer를 위한 래퍼
    결측값을 전략에 따라 대체하며 컬럼명을 보존합니다.
    
    create_missing_indicators=True로 설정하면 imputation 전 결측값 위치를 
    나타내는 indicator 컬럼들을 추가로 생성합니다.
    """
    def __init__(self, strategy: str = 'mean', columns: List[str] = None, 
                 create_missing_indicators: bool = False):
        self.strategy = strategy
        self.columns = columns
        self.create_missing_indicators = create_missing_indicators
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.missing_indicator = None
        self._input_columns = None
        if self.create_missing_indicators:
            self.missing_indicator = MissingIndicator(features='missing-only')

    def fit(self, X: pd.DataFrame, y=None):
        """Imputer 학습 및 필요시 MissingIndicator도 학습"""
        self._input_columns = list(X.columns)
        
        # Fast-fail: 전체가 NaN인 컬럼 감지
        all_null_columns = [col for col in X.columns if X[col].isnull().all()]
        if all_null_columns:
            raise ValueError(
                f"SimpleImputer는 전체가 결측값인 컬럼을 처리할 수 없습니다: {all_null_columns}\n"
                f"해당 컬럼들을 데이터에서 제거하거나 다른 전처리 방법을 사용하세요."
            )
        
        try:
            self.imputer.fit(X)
        except ValueError as e:
            error_msg = str(e)
            if "strategy" in error_msg.lower():
                raise ValueError(
                    f"SimpleImputer strategy '{self.strategy}'가 데이터 타입과 호환되지 않습니다.\n"
                    f"사용 가능한 strategy:\n"
                    f"- 'mean', 'median': 숫자형 컬럼만 가능\n"
                    f"- 'most_frequent': 모든 타입 가능\n"
                    f"- 'constant': 지정된 상수값 (fill_value 파라미터 필요)\n"
                    f"원본 오류: {error_msg}"
                ) from e
            else:
                raise
        
        # Missing indicator도 학습
        if self.create_missing_indicators and self.missing_indicator:
            self.missing_indicator.fit(X)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 Imputer를 사용하여 데이터를 변환하고 DataFrame으로 반환합니다."""
        # 1. Missing indicators 생성 (imputation 이전)
        indicator_data = None
        if self.create_missing_indicators and self.missing_indicator:
            indicator_array = self.missing_indicator.transform(X).astype(np.int64)
            
            # sklearn의 get_feature_names_out을 사용하여 실제 출력 컬럼명 확인
            try:
                indicator_feature_names = self.missing_indicator.get_feature_names_out(list(X.columns))
            except Exception:
                # 폴백: 수동으로 생성 (결측값이 있는 컬럼에 대해서만)
                indicator_feature_names = [f"missingindicator_{col}" for col in X.columns 
                                         if X[col].isnull().any()]
            
            if indicator_array.shape[1] > 0:  # indicator 컬럼이 있는 경우만
                indicator_data = pd.DataFrame(indicator_array, index=X.index, columns=indicator_feature_names)
        
        # 2. Imputation 수행
        imputed_array = self.imputer.transform(X)
        imputed_data = pd.DataFrame(imputed_array, index=X.index, columns=X.columns)
        
        # 3. 결과 결합
        if indicator_data is not None and len(indicator_data.columns) > 0:
            # Imputed data + Missing indicators
            result_data = pd.concat([imputed_data, indicator_data], axis=1)
        else:
            # Imputed data만
            result_data = imputed_data
        
        return result_data

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """출력 컬럼명 반환 (원본 + missing indicator 컬럼들)"""
        output_columns = input_columns.copy()
        
        # Missing indicator 컬럼들 추가
        if self.create_missing_indicators:
            # 결측값이 있는 컬럼들에 대한 indicator 컬럼명 생성
            for col in input_columns:
                # 실제로는 fit/transform 시점에서 결정되지만, 예상 컬럼명 반환
                output_columns.append(f"missingindicator_{col}")
        
        return output_columns
    
    def preserves_column_names(self) -> bool:
        """Missing indicator를 생성하는 경우 컬럼이 추가되므로 False"""
        return not self.create_missing_indicators
    
    def get_application_type(self) -> str:
        """SimpleImputer는 사용자가 지정한 컬럼에 적용됩니다."""
        return 'targeted'
    
    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """결측값이 있는 숫자형 컬럼만 대상으로 합니다."""
        applicable_columns = []
        for col in X.columns:
            # 숫자형이면서 결측값이 있는 컬럼만 선택
            if X[col].dtype in ['int64', 'float64'] and X[col].isnull().any():
                applicable_columns.append(col)
        return applicable_columns

PreprocessorStepRegistry.register("simple_imputer", SimpleImputerWrapper) 