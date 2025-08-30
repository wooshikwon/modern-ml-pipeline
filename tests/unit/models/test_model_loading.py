"""모델 로딩 핵심 로직 테스트 - Factory 패턴 기반

Phase 3: 핵심 모듈 테스트 확장
- Factory를 통한 동적 모델 로딩 검증
- sklearn/커스텀 모델 인터페이스 계약 준수
- Blueprint 원칙 기반 테스트 설계
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.engine.factory import Factory
from src.settings import Settings


class TestModelLoading:
    """모델 동적 로딩 핵심 기능 테스트"""
    
    def test_sklearn_model_loading_success(self, test_factories):
        """sklearn 모델 정상 로딩 테스트"""
        # Given: 분류 모델 설정 (Factory 패턴)
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When: 모델 생성
        model = factory.create_model()
        
        # Then: sklearn 호환 인터페이스 검증
        assert model is not None
        assert hasattr(model, 'fit'), "모델이 fit 메서드를 가져야 함"
        assert hasattr(model, 'predict'), "모델이 predict 메서드를 가져야 함"
        assert callable(getattr(model, 'fit')), "fit이 호출 가능해야 함"
        assert callable(getattr(model, 'predict')), "predict가 호출 가능해야 함"
        
        # Factory로 생성된 모델의 타입 검증
        assert str(type(model).__name__) == "RandomForestClassifier"
    
    def test_model_loading_with_hyperparameters(self, test_factories):
        """하이퍼파라미터 포함 모델 로딩 테스트"""
        # Given: 커스텀 하이퍼파라미터 설정
        settings_dict = test_factories['settings'].create_classification_settings("test")
        # 하이퍼파라미터 오버라이드
        settings_dict['recipe']['model']['hyperparameters'] = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When: 모델 생성
        model = factory.create_model()
        
        # Then: 하이퍼파라미터가 정확히 적용되었는지 검증
        assert model is not None
        assert model.n_estimators == 100
        assert model.max_depth == 15
        assert model.min_samples_split == 5
        assert model.random_state == 42
    
    def test_regression_model_loading(self, test_factories):
        """회귀 모델 로딩 테스트"""
        # Given: 회귀 모델 설정
        settings_dict = test_factories['settings'].create_regression_settings("test")
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When: 회귀 모델 생성
        model = factory.create_model()
        
        # Then: 회귀 모델 인터페이스 검증
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert str(type(model).__name__) == "LinearRegression"
    
    def test_invalid_model_class_path_handling(self, test_factories):
        """존재하지 않는 모델 클래스 경로 오류 처리"""
        # Given: 잘못된 클래스 경로 설정
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['class_path'] = 'non.existent.ModelClass'
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When/Then: Factory가 ValueError로 래핑한 예외 확인
        with pytest.raises(ValueError) as exc_info:
            factory.create_model()
        
        # 오류 메시지 검증 (Factory가 래핑한 메시지)
        assert "Could not load model class" in str(exc_info.value)
        assert "non.existent.ModelClass" in str(exc_info.value)
    
    def test_model_with_complex_hyperparameters(self, test_factories):
        """복잡한 하이퍼파라미터 구조 테스트"""
        # Given: 복잡한 하이퍼파라미터 포함 설정
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['class_path'] = 'sklearn.ensemble.GradientBoostingClassifier'
        settings_dict['recipe']['model']['hyperparameters'] = {
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'random_state': 42,
            'warm_start': False
        }
        
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When: 복잡한 모델 생성
        model = factory.create_model()
        
        # Then: 모든 하이퍼파라미터가 정확히 적용되었는지 검증
        assert model is not None
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert model.subsample == 0.8
        assert model.random_state == 42
        assert model.warm_start == False
        
    def test_model_interface_blueprint_compliance(self, test_factories):
        """모델 인터페이스 Blueprint 준수 검증"""
        # Given: 표준 설정으로 모델 생성
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings = Settings(**settings_dict)
        factory = Factory(settings)
        
        # When: 모델 생성 및 데이터 준비
        model = factory.create_model()
        test_data = test_factories['data'].create_classification_data(n_samples=50)
        
        # sklearn 호환을 위해 숫자형 피처만 선택
        numeric_columns = test_data.select_dtypes(include=[np.number]).columns
        X = test_data[numeric_columns].drop(['target'], axis=1, errors='ignore')
        y = test_data['target']
        
        # 최소한의 피처가 있는지 확인
        assert len(X.columns) > 0, "테스트 데이터에 숫자형 피처가 없습니다"
        
        # Then: Blueprint 인터페이스 계약 준수 검증
        # 1. 모델 학습 가능성
        fitted_model = model.fit(X, y)
        assert fitted_model is not None
        
        # 2. 예측 가능성
        predictions = model.predict(X)
        assert predictions is not None
        assert len(predictions) == len(X)
        
        # 3. 분류 모델 특성 (확률 예측 지원)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            assert probabilities is not None
            assert probabilities.shape == (len(X), len(np.unique(y)))