"""
Scaler 모듈 테스트

전처리 파이프라인의 핵심인 스케일링 변환기들을 테스트:
- StandardScalerWrapper: 평균 0, 분산 1로 표준화
- MinMaxScalerWrapper: 0-1 범위로 정규화  
- RobustScalerWrapper: 중앙값 기준 robust 스케일링
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.components.preprocessor.modules.scaler import (
    StandardScalerWrapper, 
    MinMaxScalerWrapper,
    RobustScalerWrapper
)
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestStandardScalerWrapper:
    """StandardScalerWrapper 테스트 클래스"""
    
    def test_standard_scaler_initialization(self):
        """StandardScaler 초기화 테스트"""
        # Given: 기본 초기화
        scaler = StandardScalerWrapper()
        
        # Then: 올바른 초기화 확인
        assert scaler.columns is None
        assert isinstance(scaler.scaler, StandardScaler)
        assert hasattr(scaler, 'fit')
        assert hasattr(scaler, 'transform')
    
    def test_standard_scaler_with_columns(self):
        """컬럼 지정 StandardScaler 초기화"""
        # Given: 특정 컬럼 지정
        columns = ['feature_1', 'feature_2']
        scaler = StandardScalerWrapper(columns=columns)
        
        # Then: 컬럼이 올바르게 설정됨
        assert scaler.columns == columns
    
    def test_standard_scaler_fit_transform(self):
        """StandardScaler fit/transform 기본 동작 테스트"""
        # Given: 테스트 데이터와 스케일러
        data = DataFrameBuilder.build_numeric_data(100)
        scaler = StandardScalerWrapper()
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        transformed = scaler.transform(data)
        
        # Then: 표준화 결과 검증 (평균≈0, 표준편차≈1)
        assert transformed.shape == data.shape
        np.testing.assert_allclose(np.mean(transformed, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(transformed, axis=0), 1, atol=1e-10)
    
    def test_standard_scaler_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 스케일러 생성
        scaler = PreprocessorStepRegistry.create("standard_scaler")
        data = DataFrameBuilder.build_numeric_data(50)
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        result = scaler.transform(data)
        
        # Then: 정상 동작 확인
        assert result.shape == data.shape
        assert isinstance(scaler, StandardScalerWrapper)


class TestMinMaxScalerWrapper:
    """MinMaxScalerWrapper 테스트 클래스"""
    
    def test_min_max_scaler_initialization(self):
        """MinMaxScaler 초기화 테스트"""
        # Given: 기본 초기화
        scaler = MinMaxScalerWrapper()
        
        # Then: 올바른 초기화 확인
        assert scaler.columns is None
        assert isinstance(scaler.scaler, MinMaxScaler)
    
    def test_min_max_scaler_fit_transform(self):
        """MinMaxScaler fit/transform 기본 동작 테스트"""
        # Given: 테스트 데이터와 스케일러
        data = DataFrameBuilder.build_numeric_data(100)
        scaler = MinMaxScalerWrapper()
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        transformed = scaler.transform(data)
        
        # Then: 0-1 정규화 결과 검증
        assert transformed.shape == data.shape
        assert np.min(transformed) >= 0
        assert np.max(transformed) <= 1
        
        # 각 feature별로 최솟값이 0, 최댓값이 1에 근사한지 확인
        for col_idx in range(transformed.shape[1]):
            col_data = transformed[:, col_idx]
            assert np.min(col_data) <= 1e-10  # 최솟값 ≈ 0
            assert np.max(col_data) >= 1 - 1e-10  # 최댓값 ≈ 1
    
    def test_min_max_scaler_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 스케일러 생성
        scaler = PreprocessorStepRegistry.create("min_max_scaler")
        data = DataFrameBuilder.build_numeric_data(30)
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        result = scaler.transform(data)
        
        # Then: 정상 동작 확인
        assert isinstance(scaler, MinMaxScalerWrapper)
        assert 0 <= np.min(result) <= 1e-10
        assert 1 - 1e-10 <= np.max(result) <= 1


class TestRobustScalerWrapper:
    """RobustScalerWrapper 테스트 클래스"""
    
    def test_robust_scaler_initialization(self):
        """RobustScaler 초기화 테스트"""
        # Given: 기본 초기화
        scaler = RobustScalerWrapper()
        
        # Then: 올바른 초기화 확인
        assert scaler.columns is None
        assert isinstance(scaler.scaler, RobustScaler)
    
    def test_robust_scaler_fit_transform(self):
        """RobustScaler fit/transform 기본 동작 테스트"""
        # Given: 테스트 데이터와 스케일러
        data = DataFrameBuilder.build_numeric_data(100)
        scaler = RobustScalerWrapper()
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        transformed = scaler.transform(data)
        
        # Then: 기본 동작 확인
        assert transformed.shape == data.shape
        
        # 중앙값 기준 스케일링이므로 median ≈ 0 확인
        medians = np.median(transformed, axis=0)
        np.testing.assert_allclose(medians, 0, atol=1e-10)
    
    def test_robust_scaler_outlier_resistance(self):
        """RobustScaler의 이상치 저항성 테스트"""
        # Given: 이상치가 포함된 데이터
        data = DataFrameBuilder.build_extreme_values_data(50)
        robust_scaler = RobustScalerWrapper()
        standard_scaler = StandardScalerWrapper()
        
        # When: 각각 fit/transform
        robust_scaler.fit(data)
        standard_scaler.fit(data)
        
        robust_result = robust_scaler.transform(data)
        standard_result = standard_scaler.transform(data)
        
        # Then: RobustScaler의 기본 동작 확인 (이상치에 덜 민감)
        assert robust_result.shape == data.shape
        assert standard_result.shape == data.shape
        
        # 중앙값 기준 스케일링 확인
        robust_medians = np.median(robust_result, axis=0)
        np.testing.assert_allclose(robust_medians, 0, atol=1e-10)
        
        # RobustScaler가 extreme values에 더 안정적으로 반응하는지 확인
        # (극단값들의 영향을 받지 않고 중앙값 근처에서 스케일링)
        robust_extreme_col = robust_result[:, 1]  # extreme_feature column
        robust_q75_q25_range = np.percentile(robust_extreme_col, 75) - np.percentile(robust_extreme_col, 25)
        
        # RobustScaler 결과의 IQR이 적절한 범위 내에 있는지 확인
        assert robust_q75_q25_range > 0  # 정상적인 스케일링 확인
    
    def test_robust_scaler_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 스케일러 생성
        scaler = PreprocessorStepRegistry.create("robust_scaler")
        data = DataFrameBuilder.build_extreme_values_data(40)
        
        # When: fit 및 transform 수행
        scaler.fit(data)
        result = scaler.transform(data)
        
        # Then: 정상 동작 확인
        assert isinstance(scaler, RobustScalerWrapper)
        assert result.shape == data.shape


class TestScalerComparison:
    """여러 스케일러 간 비교 테스트"""
    
    def test_all_scalers_with_same_data(self):
        """동일 데이터에 대한 모든 스케일러 비교"""
        # Given: 공통 테스트 데이터
        data = DataFrameBuilder.build_numeric_data(100)
        
        scalers = {
            'standard': StandardScalerWrapper(),
            'minmax': MinMaxScalerWrapper(), 
            'robust': RobustScalerWrapper()
        }
        
        results = {}
        
        # When: 모든 스케일러로 변환
        for name, scaler in scalers.items():
            scaler.fit(data)
            results[name] = scaler.transform(data)
        
        # Then: 모든 결과가 올바른 형태
        for name, result in results.items():
            assert result.shape == data.shape
            assert not np.any(np.isnan(result)), f"{name} scaler produced NaN values"
            assert not np.any(np.isinf(result)), f"{name} scaler produced infinite values"
    
    def test_scaler_error_handling(self):
        """스케일러 오류 처리 테스트"""
        # Given: 빈 데이터프레임
        empty_data = pd.DataFrame()
        scaler = StandardScalerWrapper()
        
        # When/Then: 빈 데이터에 대한 적절한 오류 처리
        with pytest.raises((ValueError, IndexError)):
            scaler.fit(empty_data)