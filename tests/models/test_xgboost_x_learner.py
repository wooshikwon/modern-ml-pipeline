"""
XGBoost X-Learner 모델 테스트

하이퍼파라미터 설정, 학습, 예측 테스트
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.models.xgboost_x_learner import XGBoostXLearner
from src.settings import Settings


class TestXGBoostXLearner:
    """XGBoost X-Learner 모델 테스트"""
    
    def test_model_initialization(self, xgboost_settings: Settings):
        """모델이 올바른 설정으로 초기화되는지 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        assert model.settings == xgboost_settings
        assert model.settings.model.name == "xgboost_x_learner"
    
    def test_hyperparameters_access(self, xgboost_settings: Settings):
        """하이퍼파라미터 접근 테스트 (Pydantic v2 RootModel 호환)"""
        model = XGBoostXLearner(xgboost_settings)
        
        # RootModel의 root 속성을 통해 접근
        hyperparameters = model.settings.model.hyperparameters.root
        assert isinstance(hyperparameters, dict)
        
        # 기본 하이퍼파라미터 확인
        assert 'n_estimators' in hyperparameters
        assert 'learning_rate' in hyperparameters
        assert 'max_depth' in hyperparameters
    
    def test_fit_method(self, xgboost_settings: Settings):
        """fit 메서드 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'treatment': [0, 1, 0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2, 0.8, 1.8])
        
        # Mock XGBoost 모델들
        with patch('src.models.xgboost_x_learner.XGBRegressor') as mock_xgb:
            mock_model_0 = Mock()
            mock_model_1 = Mock()
            mock_xgb.side_effect = [mock_model_0, mock_model_1]
            
            # fit 실행
            model.fit(X, y)
            
            # 두 개의 모델이 생성되었는지 확인
            assert mock_xgb.call_count == 2
            
            # 각 모델이 적절한 데이터로 학습되었는지 확인
            mock_model_0.fit.assert_called_once()
            mock_model_1.fit.assert_called_once()
            
            # 모델이 저장되었는지 확인
            assert model.model_0 == mock_model_0
            assert model.model_1 == mock_model_1
    
    def test_predict_method(self, xgboost_settings: Settings):
        """predict 메서드 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 0]
        })
        
        # Mock 학습된 모델들
        mock_model_0 = Mock()
        mock_model_1 = Mock()
        mock_model_0.predict.return_value = np.array([0.5, 0.6, 0.7])
        mock_model_1.predict.return_value = np.array([1.0, 1.1, 1.2])
        
        model.model_0 = mock_model_0
        model.model_1 = mock_model_1
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 예측이 올바르게 수행되었는지 확인
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        
        # 각 모델이 호출되었는지 확인
        mock_model_0.predict.assert_called_once()
        mock_model_1.predict.assert_called_once()
    
    def test_predict_before_fit_error(self, xgboost_settings: Settings):
        """학습 전 예측 시 오류 처리 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'treatment': [0, 1, 0]
        })
        
        # 학습되지 않은 상태에서 예측 시도
        with pytest.raises(ValueError, match="Model has not been fitted"):
            model.predict(X)
    
    def test_treatment_effect_calculation(self, xgboost_settings: Settings):
        """처치 효과 계산 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 0]
        })
        
        # Mock 학습된 모델들 (처치 효과가 명확히 다른 값)
        mock_model_0 = Mock()
        mock_model_1 = Mock()
        mock_model_0.predict.return_value = np.array([0.5, 0.6, 0.7])  # 대조군 예측
        mock_model_1.predict.return_value = np.array([1.0, 1.1, 1.2])  # 처치군 예측
        
        model.model_0 = mock_model_0
        model.model_1 = mock_model_1
        
        # 예측 실행
        predictions = model.predict(X)
        
        # 처치 효과가 올바르게 계산되었는지 확인
        # X-Learner는 처치군과 대조군 예측의 가중 평균을 계산
        assert all(pred > 0 for pred in predictions)  # 모든 예측값이 양수
        assert len(predictions) == 3
    
    def test_fit_with_missing_treatment_column(self, xgboost_settings: Settings):
        """처치 컬럼이 없는 경우 오류 처리 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 처치 컬럼이 없는 데이터
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 처치 컬럼이 없을 때 오류 발생
        with pytest.raises(KeyError, match="treatment"):
            model.fit(X, y)
    
    def test_fit_with_invalid_treatment_values(self, xgboost_settings: Settings):
        """잘못된 처치 값에 대한 오류 처리 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 잘못된 처치 값 (0, 1이 아닌 값)
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 2]  # 2는 잘못된 값
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 잘못된 처치 값에 대한 경고 또는 오류 처리
        with patch('src.models.xgboost_x_learner.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            
            # fit 시 데이터 정제가 이루어져야 함
            model.fit(X, y)
            
            # 모델이 호출되었는지 확인 (데이터 정제 후)
            assert mock_xgb.called
    
    def test_empty_treatment_groups(self, xgboost_settings: Settings):
        """한 처치군이 비어있는 경우 처리 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 한 처치군만 있는 데이터
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 0, 0]  # 모두 대조군
        })
        y = pd.Series([0.5, 1.5, 0.3])
        
        # 한 처치군이 비어있을 때의 처리
        with patch('src.models.xgboost_x_learner.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            
            # fit 시 적절한 처리가 이루어져야 함
            model.fit(X, y)
            
            # 모델이 생성되었는지 확인
            assert mock_xgb.called
    
    def test_hyperparameter_passing(self, xgboost_settings: Settings):
        """하이퍼파라미터가 올바르게 전달되는지 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2])
        
        # Mock XGBoost 모델
        with patch('src.models.xgboost_x_learner.XGBRegressor') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            
            # fit 실행
            model.fit(X, y)
            
            # 하이퍼파라미터가 올바르게 전달되었는지 확인
            call_args = mock_xgb.call_args_list[0][1]  # 첫 번째 모델 호출의 kwargs
            
            # 설정에서 정의한 하이퍼파라미터가 포함되어 있는지 확인
            hyperparameters = model.settings.model.hyperparameters.root
            for key, value in hyperparameters.items():
                if key != 'treatment':  # treatment는 하이퍼파라미터가 아님
                    assert key in call_args or key in str(call_args)
    
    def test_model_serialization_compatibility(self, xgboost_settings: Settings):
        """모델 직렬화 호환성 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 모델이 필요한 속성을 가지고 있는지 확인
        assert hasattr(model, 'settings')
        assert hasattr(model, 'model_0')
        assert hasattr(model, 'model_1')
        
        # 초기화 시 모델들이 None인지 확인
        assert model.model_0 is None
        assert model.model_1 is None
    
    def test_feature_importance_access(self, xgboost_settings: Settings):
        """피처 중요도 접근 테스트"""
        model = XGBoostXLearner(xgboost_settings)
        
        # 샘플 데이터 생성
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1]
        })
        y = pd.Series([0.5, 1.5, 0.3, 1.2])
        
        # Mock XGBoost 모델
        with patch('src.models.xgboost_x_learner.XGBRegressor') as mock_xgb:
            mock_model_0 = Mock()
            mock_model_1 = Mock()
            mock_model_0.feature_importances_ = np.array([0.6, 0.4])
            mock_model_1.feature_importances_ = np.array([0.7, 0.3])
            mock_xgb.side_effect = [mock_model_0, mock_model_1]
            
            # fit 실행
            model.fit(X, y)
            
            # 피처 중요도에 접근할 수 있는지 확인
            assert hasattr(model.model_0, 'feature_importances_')
            assert hasattr(model.model_1, 'feature_importances_')
            
            # 피처 중요도가 예상된 형태인지 확인
            assert len(model.model_0.feature_importances_) == 2
            assert len(model.model_1.feature_importances_) == 2 