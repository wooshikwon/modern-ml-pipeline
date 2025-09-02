"""MockComponentRegistry 검증 테스트"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from tests.factories.test_data_factory import TestDataFactory
from tests.factories.settings_factory import SettingsFactory
from tests.mocks.component_registry import MockComponentRegistry


@pytest.mark.core
@pytest.mark.unit
class TestMockComponentRegistry:
    """MockComponentRegistry 기능 검증"""
    
    def test_get_augmenter_basic(self):
        """Augmenter Mock 기본 기능 검증"""
        augmenter = MockComponentRegistry.get_augmenter("pass_through")
        
        assert augmenter is not None
        assert hasattr(augmenter, 'augment')
        assert callable(augmenter.augment)
    
    def test_augmenter_blueprint_contract(self):
        """Augmenter Mock Blueprint 계약 준수 검증"""
        augmenter = MockComponentRegistry.get_augmenter("pass_through")
        
        # 테스트 데이터로 augment 호출
        test_data = TestDataFactory.create_minimal_entity_data(3)
        result = augmenter.augment(test_data, run_mode="train")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        
        # Entity 스키마 보존 확인
        assert 'user_id' in result.columns
        # timestamp 컬럼 존재 (event_timestamp 또는 event_ts)
        has_timestamp = any(col in result.columns for col in ['event_timestamp', 'event_ts'])
        assert has_timestamp
    
    def test_get_preprocessor_sklearn_interface(self):
        """Preprocessor Mock sklearn 인터페이스 검증"""
        preprocessor = MockComponentRegistry.get_preprocessor("simple_scaler")
        
        assert hasattr(preprocessor, 'fit')
        assert hasattr(preprocessor, 'transform')  
        assert hasattr(preprocessor, 'fit_transform')
        
        # 인터페이스 호출 테스트
        test_data = TestDataFactory.create_classification_data(10)
        X = test_data.drop(['target'], axis=1)
        
        # fit_transform 호출
        result = preprocessor.fit_transform(X)
        assert result is not None
        
        # fit 후 transform 호출
        preprocessor.fit(X)
        result2 = preprocessor.transform(X)
        assert result2 is not None
    
    def test_get_model_sklearn_interface(self):
        """Model Mock sklearn 인터페이스 검증"""
        classifier = MockComponentRegistry.get_model("classifier")
        
        assert hasattr(classifier, 'fit')
        assert hasattr(classifier, 'predict')
        assert hasattr(classifier, 'predict_proba')
        
        # 분류기 테스트
        test_data = TestDataFactory.create_classification_data(20)
        X = test_data.drop(['target'], axis=1).select_dtypes(include=[np.number])
        y = test_data['target']
        
        # fit 호출
        classifier.fit(X, y)
        
        # predict 호출
        predictions = classifier.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
        # predict_proba 호출 (분류기 전용)
        probabilities = classifier.predict_proba(X)
        assert probabilities.shape[0] == len(X)
        assert probabilities.shape[1] == 2  # 2클래스
    
    def test_get_model_regressor(self):
        """Regressor Mock 검증"""
        regressor = MockComponentRegistry.get_model("regressor")
        
        assert hasattr(regressor, 'fit')
        assert hasattr(regressor, 'predict')
        # 회귀기는 predict_proba가 없음
        
        test_data = TestDataFactory.create_regression_data(15)
        X = test_data.drop(['target'], axis=1).select_dtypes(include=[np.number])
        y = test_data['target']
        
        regressor.fit(X, y)
        predictions = regressor.predict(X)
        
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
    
    def test_get_factory_comprehensive(self, test_factories):
        """Factory Mock 종합 검증"""
        settings_dict = test_factories['settings'].create_classification_settings()
        factory = test_factories['mocks'].get_factory(settings_dict)
        
        # Factory 인터페이스 확인
        expected_methods = [
            'create_augmenter', 'create_preprocessor', 'create_model', 
            'create_evaluator', 'create_data_adapter'
        ]
        
        for method_name in expected_methods:
            assert hasattr(factory, method_name)
            assert callable(getattr(factory, method_name))
        
        # Settings 속성 확인
        assert hasattr(factory, 'settings')
        assert factory.settings is not None
        
        # model_config 속성 확인
        assert hasattr(factory, 'model_config')
    
    def test_factory_component_creation(self, test_factories):
        """Factory Mock 컴포넌트 생성 검증"""
        settings_dict = test_factories['settings'].create_classification_settings()
        factory = test_factories['mocks'].get_factory(settings_dict)
        
        # 각 컴포넌트 생성
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        evaluator = factory.create_evaluator()
        adapter = factory.create_data_adapter()
        
        # 기본 인터페이스 확인
        assert hasattr(augmenter, 'augment')
        assert hasattr(preprocessor, 'fit_transform')
        assert hasattr(model, 'predict')
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(adapter, 'read')
    
    def test_factory_serving_mode_restriction(self, test_factories):
        """Factory Mock Serving 모드 제약 검증"""
        settings_dict = test_factories['settings'].create_classification_settings()
        factory = test_factories['mocks'].get_factory(settings_dict)
        
        # Serving 모드에서는 pass_through augmenter 사용 금지
        with pytest.raises(TypeError, match="Serving에서는.*금지"):
            factory.create_augmenter(run_mode="serving")
    
    def test_settings_mock_nested_access(self, test_factories):
        """Settings Mock 중첩 접근 검증"""
        settings_dict = test_factories['settings'].create_classification_settings()
        settings_mock = test_factories['mocks'].get_settings(settings_dict)
        
        # 중첩된 속성 접근
        assert hasattr(settings_mock, 'environment')
        assert hasattr(settings_mock.environment, 'env_name')
        assert settings_mock.environment.env_name == 'test'
        
        assert hasattr(settings_mock, 'recipe')
        assert hasattr(settings_mock.recipe, 'model')
        assert hasattr(settings_mock.recipe.model, 'data_interface')
        assert settings_mock.recipe.model.data_interface.task_type == 'classification'
    
    def test_data_adapter_mock(self):
        """데이터 어댑터 Mock 검증"""
        adapter = MockComponentRegistry.get_data_adapter("storage")
        
        assert hasattr(adapter, 'read')
        assert hasattr(adapter, 'write')
        
        # read 호출
        result = adapter.read("dummy_source")
        assert isinstance(result, pd.DataFrame)
        assert 'user_id' in result.columns
        assert 'target' in result.columns
        
        # write 호출
        test_data = TestDataFactory.create_classification_data(5)
        write_result = adapter.write(test_data, "dummy_destination")
        assert 'status' in write_result
        assert write_result['status'] == 'success'
    
    def test_registry_caching(self):
        """Registry 캐싱 기능 검증"""
        # 동일한 타입 요청 시 같은 인스턴스 반환
        augmenter1 = MockComponentRegistry.get_augmenter("pass_through")
        augmenter2 = MockComponentRegistry.get_augmenter("pass_through")
        
        assert augmenter1 is augmenter2  # 동일한 객체
        
        # 다른 타입은 다른 인스턴스
        augmenter3 = MockComponentRegistry.get_augmenter("feature_store")
        assert augmenter1 is not augmenter3
    
    def test_registry_reset(self):
        """Registry 리셋 기능 검증"""
        # Mock 생성
        augmenter = MockComponentRegistry.get_augmenter("pass_through")
        assert augmenter is not None
        
        # 캐시 상태 확인
        stats_before = MockComponentRegistry.get_cache_stats()
        assert stats_before['total_cached_instances'] > 0
        
        # 리셋
        MockComponentRegistry.reset_all()
        
        # 리셋 후 캐시 확인
        stats_after = MockComponentRegistry.get_cache_stats()
        assert stats_after['total_cached_instances'] == 0
        
        # 새로 생성된 인스턴스는 기존과 다름
        augmenter_new = MockComponentRegistry.get_augmenter("pass_through")
        assert augmenter is not augmenter_new
    
    def test_cache_stats(self):
        """캐시 통계 기능 검증"""
        # 초기 상태
        MockComponentRegistry.reset_all()
        
        # 다양한 Mock 생성
        MockComponentRegistry.get_augmenter("pass_through")
        MockComponentRegistry.get_preprocessor("simple_scaler")
        MockComponentRegistry.get_model("classifier")
        
        stats = MockComponentRegistry.get_cache_stats()
        
        assert stats['total_cached_instances'] >= 3
        assert 'augmenter' in stats['cached_types']
        assert 'preprocessor' in stats['cached_types']
        assert 'model' in stats['cached_types']
        assert len(stats['cache_keys']) >= 3
    
    @pytest.mark.extended  
    def test_mock_contract_consistency(self):
        """Mock 계약 일관성 검증 - 실제 인터페이스와 유사해야 함"""
        # 이 테스트는 실제 컴포넌트와 Mock의 인터페이스 일치성을 확인
        # Contract Testing의 기반이 됨
        
        augmenter = MockComponentRegistry.get_augmenter("pass_through")
        preprocessor = MockComponentRegistry.get_preprocessor("simple_scaler")
        model = MockComponentRegistry.get_model("classifier")
        
        # 예상 인터페이스 검증
        augmenter_methods = ['augment']
        preprocessor_methods = ['fit', 'transform', 'fit_transform']
        model_methods = ['fit', 'predict', 'predict_proba']
        
        for method in augmenter_methods:
            assert hasattr(augmenter, method)
            
        for method in preprocessor_methods:
            assert hasattr(preprocessor, method)
            
        for method in model_methods:
            assert hasattr(model, method)