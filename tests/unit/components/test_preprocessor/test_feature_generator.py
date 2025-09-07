"""
Feature Generator 모듈 테스트

새로운 피처 생성 변환기들을 테스트:
- TreeBasedFeatureGenerator: Random Forest 기반 피처 생성 (supervised)
- PolynomialFeaturesWrapper: 다항식 피처 생성 (unsupervised)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

from src.components.preprocessor.modules.feature_generator import (
    TreeBasedFeatureGenerator,
    PolynomialFeaturesWrapper
)
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestTreeBasedFeatureGenerator:
    """TreeBasedFeatureGenerator 테스트 클래스 (supervised)"""
    
    def test_tree_based_generator_initialization(self):
        """TreeBasedFeatureGenerator 초기화 테스트"""
        # Given: 기본 초기화
        generator = TreeBasedFeatureGenerator()
        
        # Then: 올바른 초기화 확인
        assert generator.n_estimators == 10
        assert generator.max_depth == 3
        assert generator.random_state == 42
        assert generator.columns is None
        assert isinstance(generator.tree_model_, RandomForestClassifier)
        assert hasattr(generator, 'fit')
        assert hasattr(generator, 'transform')
    
    def test_tree_based_generator_with_parameters(self):
        """파라미터 지정 TreeBasedFeatureGenerator 초기화"""
        # Given: 파라미터 지정
        n_estimators = 20
        max_depth = 5
        random_state = 123
        columns = ['feature_1', 'feature_2']
        
        generator = TreeBasedFeatureGenerator(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            columns=columns
        )
        
        # Then: 파라미터가 올바르게 설정됨
        assert generator.n_estimators == n_estimators
        assert generator.max_depth == max_depth
        assert generator.random_state == random_state
        assert generator.columns == columns
    
    def test_tree_based_generator_fit_transform_with_target(self):
        """TreeBasedFeatureGenerator fit/transform 타겟 변수 포함 테스트"""
        # Given: 피처 생성용 데이터와 타겟 변수
        data = DataFrameBuilder.build_feature_generation_data(100)
        X = data[['feature_1', 'feature_2', 'feature_3']]
        y = data['target']
        
        generator = TreeBasedFeatureGenerator(n_estimators=5, max_depth=2)
        
        # When: fit 및 transform 수행 (타겟 포함)
        generator.fit(X, y)
        transformed = generator.transform(X)
        
        # Then: 트리 기반 피처 생성 결과 검증
        assert transformed.shape[0] == X.shape[0]  # 행 수는 동일
        assert transformed.shape[1] > 0  # 새로운 피처들이 생성됨
        
        # 변환 결과가 binary 값인지 확인 (one-hot encoding 결과)
        unique_values = np.unique(transformed.values.ravel())
        assert set(unique_values).issubset({0, 1})  # 0과 1만 포함
        
        # 각 행은 정확히 n_estimators개의 1을 가져야 함 (각 트리마다 하나의 leaf)
        row_sums = transformed.sum(axis=1)
        assert np.all(row_sums == generator.n_estimators)
    
    def test_tree_based_generator_requires_target(self):
        """TreeBasedFeatureGenerator 타겟 변수 필수 요구사항 테스트"""
        # Given: 피처 데이터 (타겟 없음)
        data = DataFrameBuilder.build_feature_generation_data(50)
        X = data[['feature_1', 'feature_2']]
        
        generator = TreeBasedFeatureGenerator()
        
        # When/Then: 타겟 없이 fit 시도 시 오류 발생
        with pytest.raises(ValueError, match="TreeBasedFeatureGenerator requires a target variable"):
            generator.fit(X)  # y=None
    
    def test_tree_based_generator_feature_names(self):
        """TreeBasedFeatureGenerator 피처 이름 생성 테스트"""
        # Given: 작은 데이터셋
        data = DataFrameBuilder.build_feature_generation_data(30)
        X = data[['feature_1', 'feature_2']]
        y = data['target']
        
        generator = TreeBasedFeatureGenerator(n_estimators=3, max_depth=2)
        
        # When: fit 후 feature names 확인
        generator.fit(X, y)
        # TreeBasedFeatureGenerator는 실제로 transform을 통해 열이 생성됨
        transformed = generator.transform(X)
        feature_names = list(transformed.columns)
        
        # Then: 적절한 피처 이름 생성됨
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_tree_based_generator_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 생성기 생성
        generator = PreprocessorStepRegistry.create("tree_based_feature_generator")
        data = DataFrameBuilder.build_feature_generation_data(50)
        X = data[['feature_1', 'feature_2']]
        y = data['target']
        
        # When: fit 및 transform 수행
        generator.fit(X, y)
        result = generator.transform(X)
        
        # Then: 정상 동작 확인
        assert isinstance(generator, TreeBasedFeatureGenerator)
        assert result.shape[0] == X.shape[0]
        assert result.shape[1] > 0


class TestPolynomialFeaturesWrapper:
    """PolynomialFeaturesWrapper 테스트 클래스 (unsupervised)"""
    
    def test_polynomial_features_initialization_default(self):
        """PolynomialFeatures 기본 초기화 테스트"""
        # Given: 기본 초기화
        poly = PolynomialFeaturesWrapper()
        
        # Then: 올바른 초기화 확인
        assert poly.degree == 2
        assert poly.include_bias == False
        assert poly.interaction_only == False
        assert poly.columns is None
        assert isinstance(poly.poly, PolynomialFeatures)
    
    def test_polynomial_features_with_parameters(self):
        """파라미터 지정 PolynomialFeatures 초기화"""
        # Given: 파라미터 지정
        degree = 3
        include_bias = True
        interaction_only = True
        columns = ['feature_1', 'feature_2']
        
        poly = PolynomialFeaturesWrapper(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
            columns=columns
        )
        
        # Then: 파라미터가 올바르게 설정됨
        assert poly.degree == degree
        assert poly.include_bias == include_bias
        assert poly.interaction_only == interaction_only
        assert poly.columns == columns
    
    def test_polynomial_features_fit_transform(self):
        """PolynomialFeatures fit/transform 기본 동작 테스트"""
        # Given: 피처 생성용 데이터
        data = DataFrameBuilder.build_feature_generation_data(100)
        test_data = data[['feature_1', 'feature_2']]  # 2개 피처
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        # When: fit 및 transform 수행
        poly.fit(test_data)
        transformed = poly.transform(test_data)
        
        # Then: 다항식 피처 생성 결과 검증
        assert transformed.shape[0] == test_data.shape[0]  # 행 수는 동일
        # degree=2, 2개 피처 -> 1 + 2 + 1 = 5개 피처 (x1, x2, x1*x2, x1^2, x2^2)
        assert transformed.shape[1] == 5  # include_bias=False이므로 상수항 제외
        
        # 원본 피처들이 포함되어 있는지 확인 (첫 두 열)
        np.testing.assert_allclose(transformed.iloc[:, :2], test_data.values, rtol=1e-10)
    
    def test_polynomial_features_degree_3(self):
        """3차 다항식 피처 생성 테스트"""
        # Given: 작은 데이터셋
        data = DataFrameBuilder.build_feature_generation_data(20)
        test_data = data[['feature_1', 'feature_2']]
        
        poly = PolynomialFeaturesWrapper(degree=3, include_bias=False)
        
        # When: fit 및 transform 수행
        poly.fit(test_data)
        transformed = poly.transform(test_data)
        
        # Then: degree=3, 2개 피처의 경우 더 많은 피처 생성됨
        # 1, x1, x2, x1*x2, x1^2, x2^2, x1^2*x2, x1*x2^2, x1^3, x2^3 = 10개 (bias 제외하면 9개)
        assert transformed.shape[1] == 9  # bias=False
        assert transformed.shape[0] == test_data.shape[0]
    
    def test_polynomial_features_interaction_only(self):
        """interaction_only 옵션 테스트"""
        # Given: 피처 데이터
        data = DataFrameBuilder.build_feature_generation_data(50)
        test_data = data[['feature_1', 'feature_2', 'feature_3']]  # 3개 피처
        
        poly = PolynomialFeaturesWrapper(degree=2, interaction_only=True, include_bias=False)
        
        # When: fit 및 transform 수행
        poly.fit(test_data)
        transformed = poly.transform(test_data)
        
        # Then: interaction_only=True이므로 제곱항 없이 교차항만 생성
        # x1, x2, x3, x1*x2, x1*x3, x2*x3 = 6개 피처
        assert transformed.shape[1] == 6
        
        # 원본 피처들이 그대로 포함되는지 확인
        np.testing.assert_allclose(transformed.iloc[:, :3], test_data.values, rtol=1e-10)
    
    def test_polynomial_features_with_bias(self):
        """include_bias 옵션 테스트"""
        # Given: 피처 데이터
        data = DataFrameBuilder.build_feature_generation_data(30)
        test_data = data[['feature_1']]  # 1개 피처
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=True)
        
        # When: fit 및 transform 수행
        poly.fit(test_data)
        transformed = poly.transform(test_data)
        
        # Then: bias 포함으로 상수항(1) 추가
        # 1, x1, x1^2 = 3개 피처
        assert transformed.shape[1] == 3
        
        # 첫 번째 열이 모두 1인지 확인 (bias term)
        np.testing.assert_allclose(transformed.iloc[:, 0], 1.0, rtol=1e-10)
    
    def test_polynomial_features_feature_names(self):
        """PolynomialFeatures 피처 이름 생성 테스트"""
        # Given: 간단한 데이터
        data = DataFrameBuilder.build_feature_generation_data(20)
        test_data = data[['feature_1', 'feature_2']]
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        # When: fit 후 feature names 확인
        poly.fit(test_data)
        # PolynomialFeatures는 실제로 transform을 통해 열이 생성됨
        transformed = poly.transform(test_data)
        feature_names = list(transformed.columns)
        
        # Then: 적절한 피처 이름 생성됨
        assert len(feature_names) == 5  # x1, x2, x1*x2, x1^2, x2^2
        assert any('feature_1' in name for name in feature_names)
    
    def test_polynomial_features_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 생성기 생성
        poly = PreprocessorStepRegistry.create("polynomial_features")
        data = DataFrameBuilder.build_feature_generation_data(40)
        test_data = data[['feature_1', 'feature_2']]
        
        # When: fit 및 transform 수행
        poly.fit(test_data)
        result = poly.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(poly, PolynomialFeaturesWrapper)
        assert result.shape[0] == test_data.shape[0]
        assert result.shape[1] > test_data.shape[1]  # 피처 수 증가


class TestFeatureGeneratorComparison:
    """여러 피처 생성기 간 비교 테스트"""
    
    def test_different_generators_same_data(self):
        """동일 데이터에 대한 여러 생성기 비교"""
        # Given: 공통 테스트 데이터
        data = DataFrameBuilder.build_feature_generation_data(50)
        X = data[['feature_1', 'feature_2']]
        y = data['target']
        
        # When: 각 생성기로 변환
        # TreeBased (supervised)
        tree_gen = TreeBasedFeatureGenerator(n_estimators=3, max_depth=2)
        tree_gen.fit(X, y)
        tree_result = tree_gen.transform(X)
        
        # Polynomial (unsupervised)
        poly_gen = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        poly_gen.fit(X)
        poly_result = poly_gen.transform(X)
        
        # Then: 각 생성기의 특성 확인
        # TreeBased: binary features (0 또는 1)
        assert set(np.unique(tree_result.values.ravel())).issubset({0, 1})
        assert tree_result.shape[0] == X.shape[0]
        
        # Polynomial: continuous features, 더 많은 수치적 변화
        assert poly_result.shape[0] == X.shape[0]
        assert poly_result.shape[1] == 5  # x1, x2, x1*x2, x1^2, x2^2
        
        # 원본 피처들이 polynomial에서는 보존됨
        np.testing.assert_allclose(poly_result.iloc[:, :2], X.values, rtol=1e-10)


class TestFeatureGeneratorErrorHandling:
    """FeatureGenerator 오류 처리 테스트"""
    
    def test_tree_generator_with_empty_dataframe(self):
        """TreeBasedFeatureGenerator 빈 데이터프레임 처리 테스트"""
        # Given: 빈 데이터프레임
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)
        
        generator = TreeBasedFeatureGenerator()
        
        # When/Then: 빈 데이터에 대한 적절한 처리
        try:
            generator.fit(empty_X, empty_y)
            result = generator.transform(empty_X)
            # 빈 결과라도 형태는 유지되어야 함
            assert result.shape[0] == 0
        except (ValueError, IndexError):
            # sklearn이 빈 데이터를 처리하지 못하는 것은 정상적 동작
            pass
    
    def test_polynomial_generator_with_single_constant_feature(self):
        """PolynomialFeatures 상수 피처 처리 테스트"""
        # Given: 모든 값이 동일한 피처
        constant_data = pd.DataFrame({
            'constant_feature': [5.0] * 20
        })
        
        poly = PolynomialFeaturesWrapper(degree=2, include_bias=False)
        
        # When: fit 및 transform 수행
        poly.fit(constant_data)
        result = poly.transform(constant_data)
        
        # Then: 상수 피처도 적절히 처리됨
        assert result.shape[0] == constant_data.shape[0]
        # x1, x1^2 = 2개 피처 (둘 다 상수값)
        assert result.shape[1] == 2
        
        # 모든 행이 동일한 값을 가져야 함
        assert np.all(result.values == result.iloc[0, :].values)
    
    def test_custom_parameters_registry_creation(self):
        """Registry를 통한 커스텀 파라미터 생성 테스트"""
        # Given: 커스텀 파라미터로 생성기들 생성
        tree_gen = PreprocessorStepRegistry.create(
            "tree_based_feature_generator", 
            n_estimators=5, 
            max_depth=2
        )
        poly_gen = PreprocessorStepRegistry.create(
            "polynomial_features", 
            degree=3, 
            include_bias=True
        )
        
        # Then: 파라미터가 올바르게 적용됨
        assert tree_gen.n_estimators == 5
        assert tree_gen.max_depth == 2
        assert poly_gen.degree == 3
        assert poly_gen.include_bias == True