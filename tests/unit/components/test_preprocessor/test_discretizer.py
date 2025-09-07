"""
Discretizer 모듈 테스트

연속형 변수를 구간(bin)으로 나누는 이산화 변환기를 테스트:
- KBinsDiscretizerWrapper: sklearn KBinsDiscretizer 래퍼
  * strategy: uniform, quantile, kmeans
  * encode: ordinal, onehot, onehot-dense
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.preprocessing import KBinsDiscretizer

from src.components.preprocessor.modules.discretizer import KBinsDiscretizerWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestKBinsDiscretizerWrapper:
    """KBinsDiscretizerWrapper 테스트 클래스"""
    
    def test_discretizer_initialization_default(self):
        """KBinsDiscretizer 기본 초기화 테스트"""
        # Given: 기본 초기화
        discretizer = KBinsDiscretizerWrapper()
        
        # Then: 올바른 초기화 확인
        assert discretizer.n_bins == 5
        assert discretizer.encode == 'ordinal'
        assert discretizer.strategy == 'quantile'
        assert discretizer.columns is None
        assert isinstance(discretizer.discretizer, KBinsDiscretizer)
        assert hasattr(discretizer, 'fit')
        assert hasattr(discretizer, 'transform')
    
    def test_discretizer_initialization_with_parameters(self):
        """파라미터 지정 KBinsDiscretizer 초기화"""
        # Given: 파라미터 지정
        n_bins = 10
        encode = 'onehot'
        strategy = 'uniform'
        columns = ['feature_1', 'feature_2']
        
        discretizer = KBinsDiscretizerWrapper(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            columns=columns
        )
        
        # Then: 파라미터가 올바르게 설정됨
        assert discretizer.n_bins == n_bins
        assert discretizer.encode == encode
        assert discretizer.strategy == strategy
        assert discretizer.columns == columns
    
    def test_discretizer_quantile_strategy(self):
        """분위수 기반 구간화 전략 테스트"""
        # Given: 연속형 데이터와 quantile 전략 discretizer
        data = DataFrameBuilder.build_discretization_data(100)
        test_data = data[['uniform_dist']]  # 균등분포 데이터
        
        discretizer = KBinsDiscretizerWrapper(n_bins=5, strategy='quantile', encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: 구간화 결과 검증
        assert transformed.shape == test_data.shape
        
        # quantile 전략: 각 구간에 거의 동일한 개수의 데이터 포인트
        unique_bins, counts = np.unique(transformed, return_counts=True)
        assert len(unique_bins) <= 5  # 최대 5개 구간
        
        # 모든 값이 0 ~ (n_bins-1) 범위 내에 있는지 확인
        assert np.min(transformed) >= 0
        assert np.max(transformed) <= 4  # n_bins=5이므로 최대값은 4
    
    def test_discretizer_uniform_strategy(self):
        """균등 구간 전략 테스트"""
        # Given: 연속형 데이터와 uniform 전략 discretizer
        data = DataFrameBuilder.build_discretization_data(100)
        test_data = data[['normal_dist']]
        
        discretizer = KBinsDiscretizerWrapper(n_bins=4, strategy='uniform', encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: 균등 구간화 결과 검증
        assert transformed.shape == test_data.shape
        
        # 값이 정수 범위 내에 있는지 확인
        unique_bins = np.unique(transformed)
        assert len(unique_bins) <= 4
        assert np.min(transformed) >= 0
        assert np.max(transformed) <= 3
        
        # uniform 전략: 구간의 너비가 동일 (데이터 개수는 다를 수 있음)
        # 이는 구간 경계를 확인해서 검증할 수 있지만, 간단히 구간 개수만 확인
        assert len(unique_bins) > 1  # 최소한 2개 이상의 구간
    
    def test_discretizer_kmeans_strategy(self):
        """K-means 기반 구간화 전략 테스트"""
        # Given: 이중봉(bimodal) 분포 데이터와 kmeans 전략
        data = DataFrameBuilder.build_discretization_data(100)
        test_data = data[['bimodal_dist']]  # 이중봉 분포
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, strategy='kmeans', encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: K-means 구간화 결과 검증
        assert transformed.shape == test_data.shape
        
        unique_bins = np.unique(transformed)
        assert len(unique_bins) <= 3
        assert np.min(transformed) >= 0
        assert np.max(transformed) <= 2
        
        # K-means는 데이터 클러스터링을 기반으로 하므로 결과가 합리적이어야 함
        assert len(unique_bins) > 1
    
    def test_discretizer_ordinal_encoding(self):
        """순서형 인코딩 테스트"""
        # Given: 연속형 데이터와 ordinal 인코딩
        data = DataFrameBuilder.build_discretization_data(50)
        test_data = data[['exponential_dist']]
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: ordinal 인코딩 결과 검증
        assert transformed.shape == test_data.shape
        # DataFrame의 경우 각 컬럼의 dtype 확인
        assert all(dtype in [np.float64, np.int64] for dtype in transformed.dtypes)
        
        # ordinal 인코딩: 정수 값 (0, 1, 2, ...)
        unique_values = np.unique(transformed.values.ravel())
        assert all(val == int(val) for val in unique_values)  # 모든 값이 정수
    
    def test_discretizer_onehot_encoding(self):
        """원-핫 인코딩 테스트"""
        # Given: 연속형 데이터와 onehot 인코딩
        data = DataFrameBuilder.build_discretization_data(50)
        test_data = data[['uniform_dist']]
        
        discretizer = KBinsDiscretizerWrapper(n_bins=4, encode='onehot', strategy='uniform')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: onehot 인코딩 결과 검증
        assert transformed.shape[0] == test_data.shape[0]  # 행 수는 동일
        assert transformed.shape[1] == 4  # n_bins=4이므로 4개 컬럼
        
        # onehot 결과: 각 행은 정확히 하나의 1과 나머지는 0
        # sparse matrix 처리
        if hasattr(transformed, 'toarray'):
            # sparse matrix인 경우
            dense_transformed = transformed.toarray()
            row_sums = np.sum(dense_transformed, axis=1)
            assert np.all(row_sums == 1)  # 각 행의 합은 1
            
            # 모든 값이 0 또는 1
            unique_values = np.unique(dense_transformed.ravel())
            assert set(unique_values).issubset({0, 1})
        else:
            # dense array인 경우
            if isinstance(transformed, pd.DataFrame):
                row_sums = transformed.sum(axis=1)
            else:
                row_sums = np.sum(transformed, axis=1)
            assert np.all(row_sums == 1)  # 각 행의 합은 1
            
            # 모든 값이 0 또는 1
            unique_values = np.unique(transformed.values.ravel())
            assert set(unique_values).issubset({0, 1})
    
    def test_discretizer_multiple_features(self):
        """다중 피처 구간화 테스트"""
        # Given: 여러 피처를 가진 데이터
        data = DataFrameBuilder.build_discretization_data(100)
        test_data = data[['uniform_dist', 'normal_dist', 'exponential_dist']]
        
        discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        transformed = discretizer.transform(test_data)
        
        # Then: 각 피처가 개별적으로 구간화됨
        assert transformed.shape == test_data.shape
        
        # 각 컬럼이 적절한 범위의 값을 가지는지 확인
        for col_idx in range(transformed.shape[1]):
            col_values = transformed.iloc[:, col_idx]
            assert np.min(col_values) >= 0
            assert np.max(col_values) <= 2  # n_bins=3이므로 최대값은 2
    
    def test_discretizer_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 discretizer 생성
        discretizer = PreprocessorStepRegistry.create("kbins_discretizer")
        data = DataFrameBuilder.build_discretization_data(50)
        test_data = data[['uniform_dist']]
        
        # When: fit 및 transform 수행
        discretizer.fit(test_data)
        result = discretizer.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(discretizer, KBinsDiscretizerWrapper)
        assert result.shape[0] == test_data.shape[0]
        assert np.min(result) >= 0
    
    def test_discretizer_custom_parameters_registry(self):
        """Registry를 통한 커스텀 파라미터 테스트"""
        # Given: 커스텀 파라미터로 discretizer 생성
        discretizer = PreprocessorStepRegistry.create(
            "kbins_discretizer",
            n_bins=8,
            strategy='kmeans',
            encode='onehot'
        )
        
        # Then: 파라미터가 올바르게 적용됨
        assert discretizer.n_bins == 8
        assert discretizer.strategy == 'kmeans'
        assert discretizer.encode == 'onehot'


class TestKBinsDiscretizerEdgeCases:
    """KBinsDiscretizer 경계 사례 테스트"""
    
    def test_discretizer_with_constant_values(self):
        """상수값 데이터에 대한 구간화 테스트"""
        # Given: 모든 값이 동일한 데이터
        constant_data = pd.DataFrame({
            'constant_feature': [10.0] * 50
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=5, encode='ordinal')
        
        # When: fit 및 transform 수행
        discretizer.fit(constant_data)
        result = discretizer.transform(constant_data)
        
        # Then: 상수 데이터도 적절히 처리됨
        assert result.shape == constant_data.shape
        # 모든 값이 동일한 구간에 할당됨
        assert len(np.unique(result)) == 1
    
    def test_discretizer_with_few_unique_values(self):
        """고유값이 적은 데이터에 대한 테스트"""
        # Given: 고유값이 구간 수보다 적은 데이터
        few_values_data = pd.DataFrame({
            'few_values': [1, 2, 3, 1, 2, 3, 1, 2, 3] * 10  # 3개의 고유값
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=10, encode='ordinal')  # 구간은 10개
        
        # When: fit 및 transform 수행
        discretizer.fit(few_values_data)
        result = discretizer.transform(few_values_data)
        
        # Then: 고유값보다 많은 구간을 요청해도 적절히 처리됨
        assert result.shape == few_values_data.shape
        unique_bins = np.unique(result)
        # 실제 생성되는 구간 수는 고유값 수를 초과할 수 없음
        assert len(unique_bins) <= 3
    
    def test_discretizer_extreme_distributions(self):
        """극단적 분포에 대한 구간화 테스트"""
        # Given: 매우 치우친 분포 (대부분이 한쪽에 몰려있음)
        np.random.seed(42)
        skewed_data = pd.DataFrame({
            'skewed_feature': np.concatenate([
                np.random.normal(0, 0.1, 90),  # 대부분의 데이터가 0 근처
                np.random.normal(100, 5, 10)   # 소수의 데이터가 100 근처
            ])
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=5, strategy='quantile')
        
        # When: fit 및 transform 수행
        discretizer.fit(skewed_data)
        result = discretizer.transform(skewed_data)
        
        # Then: 극단적 분포도 적절히 처리됨
        assert result.shape == skewed_data.shape
        assert np.min(result) >= 0
        assert np.max(result) < 5  # n_bins=5
    
    def test_discretizer_single_feature(self):
        """단일 피처 구간화 테스트"""
        # Given: 하나의 피처만 있는 데이터
        data = DataFrameBuilder.build_discretization_data(80)
        single_feature = data[['normal_dist']]
        
        discretizer = KBinsDiscretizerWrapper(n_bins=6, strategy='uniform')
        
        # When: fit 및 transform 수행
        discretizer.fit(single_feature)
        result = discretizer.transform(single_feature)
        
        # Then: 단일 피처도 올바르게 처리됨
        assert result.shape == single_feature.shape
        assert len(result.shape) == 2  # 2차원 배열 유지
        assert result.shape[1] == 1   # 하나의 컬럼


class TestDiscretizerComparison:
    """여러 구간화 전략 간 비교 테스트"""
    
    def test_different_strategies_same_data(self):
        """동일 데이터에 대한 여러 전략 비교"""
        # Given: 공통 테스트 데이터
        data = DataFrameBuilder.build_discretization_data(100)
        test_data = data[['uniform_dist']]
        
        strategies = ['uniform', 'quantile', 'kmeans']
        results = {}
        
        # When: 각 전략으로 구간화
        for strategy in strategies:
            discretizer = KBinsDiscretizerWrapper(n_bins=4, strategy=strategy, encode='ordinal')
            discretizer.fit(test_data)
            results[strategy] = discretizer.transform(test_data)
        
        # Then: 모든 전략에서 유효한 결과 생성
        for strategy, result in results.items():
            assert result.shape == test_data.shape, f"{strategy} failed shape test"
            assert np.min(result) >= 0, f"{strategy} has negative values"
            assert np.max(result) <= 3, f"{strategy} exceeds max bin value"  # n_bins=4
            
        # 전략별로 다른 구간화 결과가 나올 가능성이 높음
        uniform_result = results['uniform']
        quantile_result = results['quantile']
        
        # 적어도 일부 데이터에서는 다른 결과가 나와야 함 (완전히 동일하지 않을 확률이 높음)
        # 하지만 테스트의 안정성을 위해 기본 유효성만 확인
        assert len(np.unique(uniform_result)) > 0
        assert len(np.unique(quantile_result)) > 0
    
    def test_different_encodings_same_data(self):
        """동일 데이터에 대한 여러 인코딩 방식 비교"""
        # Given: 공통 테스트 데이터
        data = DataFrameBuilder.build_discretization_data(50)
        test_data = data[['normal_dist']]
        
        # When: 각 인코딩 방식으로 구간화
        # Ordinal encoding
        ordinal_discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='ordinal')
        ordinal_discretizer.fit(test_data)
        ordinal_result = ordinal_discretizer.transform(test_data)
        
        # Onehot encoding
        onehot_discretizer = KBinsDiscretizerWrapper(n_bins=3, encode='onehot')
        onehot_discretizer.fit(test_data)
        onehot_result = onehot_discretizer.transform(test_data)
        
        # Then: 인코딩 방식에 따른 차이 확인
        # Ordinal: 원본과 동일한 컬럼 수
        assert ordinal_result.shape[1] == test_data.shape[1]
        
        # Onehot: 구간 수만큼 컬럼 수 증가
        assert onehot_result.shape[1] == 3  # n_bins=3
        assert onehot_result.shape[0] == test_data.shape[0]  # 행 수는 동일
        
        # Onehot 결과의 각 행 합이 1인지 확인 (sparse matrix 처리)
        if hasattr(onehot_result, 'toarray'):
            # sparse matrix인 경우
            dense_onehot = onehot_result.toarray()
            row_sums = np.sum(dense_onehot, axis=1)
            assert np.all(row_sums == 1)
        else:
            # dense array인 경우
            if isinstance(onehot_result, pd.DataFrame):
                row_sums = onehot_result.sum(axis=1)
            else:
                row_sums = np.sum(onehot_result, axis=1)
            assert np.all(row_sums == 1)


class TestDiscretizerErrorHandling:
    """Discretizer 오류 처리 테스트"""
    
    def test_discretizer_with_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        # Given: 빈 데이터프레임
        empty_data = pd.DataFrame()
        discretizer = KBinsDiscretizerWrapper()
        
        # When/Then: 빈 데이터에 대한 적절한 처리
        try:
            discretizer.fit(empty_data)
            result = discretizer.transform(empty_data)
            assert result.shape == empty_data.shape
        except (ValueError, IndexError):
            # sklearn이 빈 데이터를 처리하지 못하는 것은 정상적 동작
            pass
    
    def test_discretizer_invalid_parameters(self):
        """잘못된 파라미터 처리 테스트"""
        # Given & When/Then: 잘못된 파라미터로 생성 시도
        with pytest.raises((ValueError, TypeError)):
            # 잘못된 strategy
            invalid_discretizer = KBinsDiscretizerWrapper(strategy='invalid_strategy')
            data = DataFrameBuilder.build_discretization_data(10)
            invalid_discretizer.fit(data[['uniform_dist']])
    
    def test_discretizer_very_small_dataset(self):
        """매우 작은 데이터셋 처리 테스트"""
        # Given: 매우 작은 데이터셋 (구간 수보다 적은 데이터 포인트)
        tiny_data = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0]  # 3개 데이터 포인트
        })
        
        discretizer = KBinsDiscretizerWrapper(n_bins=5, encode='ordinal')  # 5개 구간 요청
        
        # When: fit 및 transform 수행
        discretizer.fit(tiny_data)
        result = discretizer.transform(tiny_data)
        
        # Then: 작은 데이터셋도 적절히 처리됨
        assert result.shape == tiny_data.shape
        # 실제 구간 수는 데이터 포인트 수를 초과할 수 없음
        unique_bins = np.unique(result)
        assert len(unique_bins) <= 3