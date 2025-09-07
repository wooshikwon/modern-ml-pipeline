"""
Encoder 모듈 테스트

범주형 변수 인코딩 변환기들을 테스트:
- OneHotEncoderWrapper: 원-핫 인코딩 (더미 변수)
- OrdinalEncoderWrapper: 순서형 인코딩 (정수 매핑)
- CatBoostEncoderWrapper: 타겟 기반 인코딩 (supervised)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.components.preprocessor.modules.encoder import (
    OneHotEncoderWrapper,
    OrdinalEncoderWrapper,
    CatBoostEncoderWrapper
)

from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestOneHotEncoderWrapper:
    """OneHotEncoderWrapper 테스트 클래스"""
    
    def test_one_hot_encoder_initialization(self):
        """OneHotEncoder 초기화 테스트"""
        # Given: 기본 초기화
        encoder = OneHotEncoderWrapper()
        
        # Then: 올바른 초기화 확인
        assert encoder.columns is None
        assert encoder.handle_unknown == 'ignore'
        assert encoder.sparse_output == False
        assert hasattr(encoder, 'fit')
        assert hasattr(encoder, 'transform')
    
    def test_one_hot_encoder_with_parameters(self):
        """파라미터 지정 OneHotEncoder 초기화"""
        # Given: 파라미터 지정
        columns = ['category_1', 'category_2']
        encoder = OneHotEncoderWrapper(
            columns=columns,
            handle_unknown='error',
            sparse_output=True
        )
        
        # Then: 파라미터가 올바르게 설정됨
        assert encoder.columns == columns
        assert encoder.handle_unknown == 'error'
        assert encoder.sparse_output == True
    
    def test_one_hot_encoder_fit_transform(self):
        """OneHotEncoder fit/transform 기본 동작 테스트"""
        # Given: 범주형 데이터와 인코더
        data = DataFrameBuilder.build_categorical_data(100)
        categorical_cols = ['category_low_card', 'category_medium_card']
        test_data = data[categorical_cols]
        
        encoder = OneHotEncoderWrapper()
        
        # When: fit 및 transform 수행
        encoder.fit(test_data)
        transformed = encoder.transform(test_data)
        
        # Then: 원-핫 인코딩 결과 검증
        assert transformed.shape[0] == test_data.shape[0]  # 행 수는 동일
        assert transformed.shape[1] > test_data.shape[1]   # 열 수는 증가 (원-핫 확장)
        
        # 변환 결과가 0 또는 1만 포함하는지 확인 (원-핫 특성)
        unique_values = np.unique(transformed.values)
        assert all(val in [0, 1] for val in unique_values)
    
    def test_one_hot_encoder_feature_names(self):
        """OneHotEncoder 피처 이름 생성 테스트"""
        # Given: 간단한 범주형 데이터
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        encoder = OneHotEncoderWrapper()
        
        # When: fit 후 feature names 확인
        encoder.fit(data)
        # OneHotEncoder는 실제로 transform을 통해 열이 생성됨
        transformed = encoder.transform(data)
        feature_names = list(transformed.columns)
        
        # Then: 적절한 피처 이름 생성됨
        assert len(feature_names) == 3  # A, B, C = 3개 카테고리
        assert all('category_' in name for name in feature_names)
    
    def test_one_hot_encoder_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 인코더 생성
        encoder = PreprocessorStepRegistry.create("one_hot_encoder")
        data = DataFrameBuilder.build_categorical_data(50)
        test_data = data[['category_low_card']]
        
        # When: fit 및 transform 수행
        encoder.fit(test_data)
        result = encoder.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(encoder, OneHotEncoderWrapper)
        assert result.shape[0] == test_data.shape[0]
        assert result.shape[1] >= test_data.shape[1]


class TestOrdinalEncoderWrapper:
    """OrdinalEncoderWrapper 테스트 클래스"""
    
    def test_ordinal_encoder_initialization(self):
        """OrdinalEncoder 초기화 테스트"""
        # Given: 기본 초기화
        encoder = OrdinalEncoderWrapper()
        
        # Then: 올바른 초기화 확인
        assert encoder.columns is None
        assert encoder.handle_unknown == 'use_encoded_value'
        assert encoder.unknown_value == -1
    
    def test_ordinal_encoder_fit_transform(self):
        """OrdinalEncoder fit/transform 기본 동작 테스트"""
        # Given: 범주형 데이터와 인코더
        data = DataFrameBuilder.build_categorical_data(100)
        categorical_cols = ['ordinal_feature']  # Low, Medium, High
        test_data = data[categorical_cols]
        
        encoder = OrdinalEncoderWrapper()
        
        # When: fit 및 transform 수행
        encoder.fit(test_data)
        transformed = encoder.transform(test_data)
        
        # Then: 순서형 인코딩 결과 검증
        assert transformed.shape == test_data.shape  # 형태는 동일
        
        # 변환 결과가 정수인지 확인
        unique_values = np.unique(transformed.iloc[:, 0])
        assert all(isinstance(val, (int, float)) for val in unique_values)
        
        # 고유값 개수가 원본 카테고리 개수와 일치하는지 확인
        original_categories = test_data['ordinal_feature'].unique()
        assert len(unique_values) == len(original_categories)
    
    def test_ordinal_encoder_unknown_handling(self):
        """OrdinalEncoder 미지 카테고리 처리 테스트"""
        # Given: 학습 데이터와 새로운 카테고리를 포함한 테스트 데이터
        train_data = pd.DataFrame({'category': ['A', 'B', 'C']})
        test_data = pd.DataFrame({'category': ['A', 'B', 'C', 'D']})  # 'D'는 새로운 카테고리
        
        encoder = OrdinalEncoderWrapper(unknown_value=-999)
        
        # When: 학습 후 새로운 카테고리 포함 데이터 변환
        encoder.fit(train_data)
        transformed = encoder.transform(test_data)
        
        # Then: 미지 카테고리가 지정된 값으로 처리됨
        assert -999 in transformed.iloc[:, 0].values  # unknown_value 포함
    
    def test_ordinal_encoder_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 인코더 생성
        encoder = PreprocessorStepRegistry.create("ordinal_encoder")
        data = DataFrameBuilder.build_categorical_data(50)
        test_data = data[['category_low_card']]
        
        # When: fit 및 transform 수행
        encoder.fit(test_data)
        result = encoder.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(encoder, OrdinalEncoderWrapper)
        assert result.shape == test_data.shape


class TestCatBoostEncoderWrapper:
    """CatBoostEncoderWrapper 테스트 클래스 (supervised encoder)"""
    
    def test_catboost_encoder_initialization(self):
        """CatBoostEncoder 초기화 테스트"""
        # Given: 기본 초기화
        encoder = CatBoostEncoderWrapper()
        
        # Then: 올바른 초기화 확인
        assert encoder.columns is None
        assert encoder.sigma == 0.05
        assert hasattr(encoder.encoder, 'fit')
    
    def test_catboost_encoder_with_parameters(self):
        """파라미터 지정 CatBoostEncoder 초기화"""
        # Given: 파라미터 지정
        columns = ['category_1']
        encoder = CatBoostEncoderWrapper(sigma=0.1, columns=columns)
        
        # Then: 파라미터가 올바르게 설정됨
        assert encoder.sigma == 0.1
        assert encoder.columns == columns
    
    def test_catboost_encoder_fit_transform_with_target(self):
        """CatBoostEncoder fit/transform 타겟 변수 포함 테스트"""
        # Given: 범주형 데이터와 타겟 변수
        data = DataFrameBuilder.build_categorical_data(100)
        categorical_cols = ['category_low_card', 'category_medium_card']
        X = data[categorical_cols]
        y = data['target']
        
        encoder = CatBoostEncoderWrapper()
        
        # When: fit 및 transform 수행 (타겟 포함)
        encoder.fit(X, y)
        transformed = encoder.transform(X)
        
        # Then: 타겟 기반 인코딩 결과 검증
        assert transformed.shape == X.shape  # 형태는 동일
        
        # 변환 결과가 연속형 값인지 확인 (타겟 기반 인코딩 특성)
        assert np.all(np.isfinite(transformed))  # 유한한 값들
        
        # 원본 문자열과 다른 숫자형 값으로 변환되었는지 확인
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in transformed.dtypes)
    
    def test_catboost_encoder_requires_target(self):
        """CatBoostEncoder 타겟 변수 필수 요구사항 테스트"""
        # Given: 범주형 데이터 (타겟 없음)
        data = DataFrameBuilder.build_categorical_data(50)
        X = data[['category_low_card']]
        
        encoder = CatBoostEncoderWrapper()
        
        # When/Then: 타겟 없이 fit 시도 시 오류 발생
        with pytest.raises(ValueError, match="CatBoostEncoder requires a target variable"):
            encoder.fit(X)  # y=None
    
    def test_catboost_encoder_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 인코더 생성
        encoder = PreprocessorStepRegistry.create("catboost_encoder")
        data = DataFrameBuilder.build_categorical_data(50)
        X = data[['category_medium_card']]
        y = data['target']
        
        # When: fit 및 transform 수행
        encoder.fit(X, y)
        result = encoder.transform(X)
        
        # Then: 정상 동작 확인
        assert isinstance(encoder, CatBoostEncoderWrapper)
        assert result.shape == X.shape
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)


class TestEncoderComparison:
    """여러 인코더 간 비교 테스트"""
    
    def test_all_encoders_with_same_categorical_data(self):
        """동일 범주형 데이터에 대한 모든 인코더 비교"""
        # Given: 공통 범주형 데이터
        data = DataFrameBuilder.build_categorical_data(100)
        X = data[['category_low_card']]  # 간단한 카테고리 (A, B, C)
        y = data['target']
        
        # When: 각 인코더로 변환
        # OneHot
        onehot_encoder = OneHotEncoderWrapper()
        onehot_encoder.fit(X)
        onehot_result = onehot_encoder.transform(X)
        
        # Ordinal
        ordinal_encoder = OrdinalEncoderWrapper()
        ordinal_encoder.fit(X)
        ordinal_result = ordinal_encoder.transform(X)
        
        # CatBoost (타겟 필요)
        catboost_encoder = CatBoostEncoderWrapper()
        catboost_encoder.fit(X, y)
        catboost_result = catboost_encoder.transform(X)
        
        # Then: 각 인코더의 특성 확인
        # OneHot: 열 수 증가 (더미 변수)
        assert onehot_result.shape[1] > X.shape[1]
        assert np.all((onehot_result.values == 0) | (onehot_result.values == 1))
        
        # Ordinal: 형태 동일, 정수형 값
        assert ordinal_result.shape == X.shape
        
        # CatBoost: 형태 동일, 연속형 값 
        assert catboost_result.shape == X.shape
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in catboost_result.dtypes)
    
    def test_encoder_error_handling(self):
        """인코더 오류 처리 테스트"""
        # Given: 유효하지 않은 데이터 (None 포함)
        invalid_data = pd.DataFrame({'category': [None, None, None]})
        encoder = OneHotEncoderWrapper()
        
        # When: fit 후 transform 수행 (빈 데이터는 일반적으로 처리 가능)
        encoder.fit(invalid_data)
        result = encoder.transform(invalid_data)
        
        # Then: 기본적인 형태 유지 확인
        assert result.shape[0] == invalid_data.shape[0]
    
    def test_encoder_with_mixed_data_types(self):
        """혼합 데이터 타입에 대한 인코더 테스트"""
        # Given: 숫자형 + 범주형 혼합 데이터
        data = DataFrameBuilder.build_mixed_categorical_data(100)
        categorical_cols = ['category_1', 'category_2']
        X_cat = data[categorical_cols]
        y = data['target']
        
        encoders = {
            'onehot': OneHotEncoderWrapper(),
            'ordinal': OrdinalEncoderWrapper(),
            'catboost': CatBoostEncoderWrapper()
        }
        
        # When/Then: 모든 인코더가 범주형 데이터를 적절히 처리
        for name, encoder in encoders.items():
            if name == 'catboost':
                encoder.fit(X_cat, y)  # supervised
            else:
                encoder.fit(X_cat)  # unsupervised
                
            result = encoder.transform(X_cat)
            
            # 기본 유효성 검증
            assert result.shape[0] == X_cat.shape[0]  # 행 수는 보존
            assert not result.isna().any().any()  # NaN 값 없음
            assert not np.isinf(result.values).any()  # 무한값 없음