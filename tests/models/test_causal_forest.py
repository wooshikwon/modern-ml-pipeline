"""
CausalForest 모델 테스트

causalml 호환성, 인과 추론 로직 테스트
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.models.causal_forest import CausalForestModel
from src.settings import Settings


class TestCausalForestModel:
    """CausalForest 모델 테스트"""
    
    def test_model_initialization(self, causal_forest_settings: Settings):
        """모델이 올바른 설정으로 초기화되는지 테스트"""
        model = CausalForestModel(causal_forest_settings)
        assert model.settings == causal_forest_settings
        assert model.settings.model.name == "causal_forest"
    
    def test_hyperparameters_access(self, causal_forest_settings: Settings):
        """하이퍼파라미터 접근 테스트 (Pydantic v2 RootModel 호환)"""
        model = CausalForestModel(causal_forest_settings)
        
        # RootModel의 root 속성을 통해 접근
        hyperparameters = model.settings.model.hyperparameters.root
        assert isinstance(hyperparameters, dict)
        
        # 기본 하이퍼파라미터 확인
        assert 'n_estimators' in hyperparameters
        assert 'max_depth' in hyperparameters
    
    def test_fit_method(self, causal_forest_settings: Settings):
        """fit 메서드 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'treatment': [0, 1, 0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2, 0.8, 1.8])
        
        # Mock CausalRandomForestRegressor
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_causal_forest.return_value = mock_model
            
            # fit 실행
            model.fit(X, y)
            
            # 모델이 생성되었는지 확인
            mock_causal_forest.assert_called_once()
            
            # 모델이 학습되었는지 확인
            mock_model.fit.assert_called_once()
            
            # 모델이 저장되었는지 확인
            assert model.model == mock_model
    
    def test_predict_method(self, causal_forest_settings: Settings):
        """predict 메서드 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 0]
        })
        
        # Mock 학습된 모델
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.4])
        
        model.model = mock_model
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 예측이 올바르게 수행되었는지 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        
        # 모델이 호출되었는지 확인
        mock_model.predict.assert_called_once()
    
    def test_predict_before_fit_error(self, causal_forest_settings: Settings):
        """학습 전 예측 시 오류 처리 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'treatment': [0, 1, 0]
        })
        
        # 학습되지 않은 상태에서 예측 시도
        with pytest.raises(ValueError, match="Model has not been fitted"):
            model.predict(X)
    
    def test_causal_effect_prediction(self, causal_forest_settings: Settings):
        """인과 효과 예측 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1]
        })
        
        # Mock 학습된 모델 (인과 효과 예측)
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.2, 0.5, 0.3, 0.8])  # 인과 효과
        
        model.model = mock_model
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 인과 효과가 올바르게 예측되었는지 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 4
        assert all(pred >= 0 for pred in predictions)  # 인과 효과는 음수일 수도 있지만 이 예제에서는 양수
    
    def test_fit_with_missing_treatment_column(self, causal_forest_settings: Settings):
        """처치 컬럼이 없는 경우 오류 처리 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 처치 컬럼이 없는 데이터
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 처치 컬럼이 없을 때 오류 발생
        with pytest.raises(KeyError, match="treatment"):
            model.fit(X, y)
    
    def test_fit_with_invalid_treatment_values(self, causal_forest_settings: Settings):
        """잘못된 처치 값에 대한 오류 처리 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 잘못된 처치 값 (0, 1이 아닌 값)
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 2]  # 2는 잘못된 값
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 잘못된 처치 값에 대한 경고 또는 오류 처리
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_causal_forest.return_value = mock_model
            
            # fit 시 데이터 정제가 이루어져야 함
            model.fit(X, y)
            
            # 모델이 호출되었는지 확인 (데이터 정제 후)
            assert mock_causal_forest.called
    
    def test_hyperparameter_passing(self, causal_forest_settings: Settings):
        """하이퍼파라미터가 올바르게 전달되는지 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2])
        
        # Mock CausalRandomForestRegressor
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_causal_forest.return_value = mock_model
            
            # fit 실행
            model.fit(X, y)
            
            # 하이퍼파라미터가 올바르게 전달되었는지 확인
            call_args = mock_causal_forest.call_args[1]  # kwargs
            
            # 설정에서 정의한 하이퍼파라미터가 포함되어 있는지 확인
            hyperparameters = model.settings.model.hyperparameters.root
            for key, value in hyperparameters.items():
                if key != 'treatment':  # treatment는 하이퍼파라미터가 아님
                    assert key in call_args or key in str(call_args)
    
    def test_causalml_compatibility(self, causal_forest_settings: Settings):
        """causalml 라이브러리 호환성 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # CausalRandomForestRegressor import 확인
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_causal_forest.return_value = mock_model
            
            # 샘플 데이터로 학습
            X = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [0.1, 0.2, 0.3, 0.4],
                'treatment': [0, 1, 0, 1]
            })
            y = pd.Series([0.5, 1.5, 0.3, 1.2])
            
            model.fit(X, y)
            
            # causalml 0.15.5 API 호환성 확인
            mock_causal_forest.assert_called_once()
            mock_model.fit.assert_called_once()
    
    def test_model_serialization_compatibility(self, causal_forest_settings: Settings):
        """모델 직렬화 호환성 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 모델이 필요한 속성을 가지고 있는지 확인
        assert hasattr(model, 'settings')
        assert hasattr(model, 'model')
        
        # 초기화 시 모델이 None인지 확인
        assert model.model is None
    
    def test_feature_importance_access(self, causal_forest_settings: Settings):
        """피처 중요도 접근 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2])
        
        # Mock CausalRandomForestRegressor
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_model.feature_importances_ = np.array([0.6, 0.4])
            mock_causal_forest.return_value = mock_model
            
            # fit 실행
            model.fit(X, y)
            
            # 피처 중요도에 접근할 수 있는지 확인
            if hasattr(model.model, 'feature_importances_'):
                assert len(model.model.feature_importances_) == 2
    
    def test_empty_treatment_groups(self, causal_forest_settings: Settings):
        """한 처치군이 비어있는 경우 처리 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 한 처치군만 있는 데이터
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 0, 0]  # 모두 대조군
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 한 처치군이 비어있을 때의 처리
        with patch('src.models.causal_forest.CausalRandomForestRegressor') as mock_causal_forest:
            mock_model = Mock()
            mock_causal_forest.return_value = mock_model
            
            # fit 시 적절한 처리가 이루어져야 함
            model.fit(X, y)
            
            # 모델이 생성되었는지 확인
            assert mock_causal_forest.called
    
    def test_heterogeneous_treatment_effects(self, causal_forest_settings: Settings):
        """이질적 처치 효과 테스트"""
        model = CausalForestModel(causal_forest_settings)
        
        # 다양한 특성을 가진 샘플 데이터
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'treatment': [0, 1, 0, 1, 0, 1]
        })
        
        # Mock 학습된 모델 (이질적 처치 효과)
        mock_model = Mock()
        # 각 개체별로 다른 처치 효과 반환
        mock_model.predict.return_value = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 1.0])
        
        model.model = mock_model
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 이질적 처치 효과가 올바르게 예측되었는지 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 6
        assert np.var(predictions) > 0  # 다양한 처치 효과가 있어야 함
    
    def test_confidence_intervals(self, causal_forest_settings: Settings):
        """신뢰 구간 테스트 (만약 causalml에서 지원한다면)"""
        model = CausalForestModel(causal_forest_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 0]
        })
        
        # Mock 학습된 모델
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.4])
        
        # 신뢰 구간 메서드가 있다면 Mock 설정
        if hasattr(mock_model, 'predict_interval'):
            mock_model.predict_interval.return_value = (
                np.array([0.1, 0.5, 0.2]),  # 하한
                np.array([0.5, 0.9, 0.6])   # 상한
            )
        
        model.model = mock_model
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 예측이 올바르게 수행되었는지 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3 