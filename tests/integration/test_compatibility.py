"""
Blueprint v17.0 호환성 보장 통합 테스트

기존 코드와 새로운 기능이 함께 정상 동작하는지 검증하는 테스트
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.settings import Settings
from src.core.trainer import Trainer
from src.core.factory import Factory


class TestBlueprintV17Compatibility:
    """Blueprint v17.0 전체 호환성 테스트"""
    
    def test_existing_workflow_unchanged(self, xgboost_settings: Settings):
        """기존 워크플로우가 변경 없이 동작하는지 테스트"""
        # 기존 설정 그대로 사용 (hyperparameter_tuning, feature_store 없음)
        assert xgboost_settings.hyperparameter_tuning is None
        assert xgboost_settings.feature_store is None
        assert xgboost_settings.model.hyperparameter_tuning is None
        
        # 기존 컴포넌트들이 정상 동작하는지 확인
        factory = Factory(xgboost_settings)
        trainer = Trainer(xgboost_settings)
        
        # 기존 어댑터들 생성 가능
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        
        # 모든 컴포넌트가 올바른 타입인지 확인
        from src.core.augmenter import Augmenter
        from src.core.preprocessor import Preprocessor
        assert isinstance(augmenter, Augmenter)
        assert isinstance(preprocessor, Preprocessor)
        assert trainer.settings == xgboost_settings
    
    @patch('src.core.factory.Factory.create_evaluator')
    @patch('src.core.factory.Factory.create_preprocessor')
    def test_existing_training_produces_compatible_results(self, mock_preprocessor, mock_evaluator, xgboost_settings: Settings):
        """기존 학습 방식이 호환되는 결과를 생성하는지 테스트"""
        # Mock 설정
        mock_preprocessor_instance = Mock()
        mock_evaluator_instance = Mock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate.return_value = {"accuracy": 0.85}
        
        trainer = Trainer(xgboost_settings)
        
        # 샘플 데이터
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'outcome': [0, 1, 0, 1, 0, 1]
        })
        
        # 기존 방식 학습 (최적화 비활성화)
        with patch.object(trainer, '_prepare_training_data') as mock_prepare:
            mock_prepare.return_value = (sample_data[['feature1', 'feature2']], sample_data['outcome'], {})
            
            with patch.object(trainer, '_fit_model'):
                with patch.object(trainer, '_split_data') as mock_split:
                    train_data = sample_data.iloc[:4]
                    test_data = sample_data.iloc[4:]
                    mock_split.return_value = (train_data, test_data)
                    
                    # 학습 실행
                    mock_model = Mock()
                    result = trainer.train(sample_data, mock_model)
                    
                    # 결과 구조 확인 (Blueprint v17.0 확장 포함)
                    assert len(result) == 3  # preprocessor, model, training_results
                    training_results = result[2]
                    
                    # 기존 metrics 유지
                    assert "metrics" in training_results
                    
                    # 새로운 메타데이터 포함 확인 (기본값으로)
                    assert "hyperparameter_optimization" in training_results
                    assert training_results["hyperparameter_optimization"]["enabled"] is False
                    assert "training_methodology" in training_results
    
    @patch('src.core.factory.Path')
    def test_existing_pyfunc_wrapper_creation(self, mock_path, xgboost_settings: Settings):
        """기존 PyfuncWrapper 생성 방식이 호환되는지 테스트"""
        # Mock 설정
        mock_sql_file = Mock()
        mock_sql_file.read_text.return_value = "SELECT user_id, feature1 FROM table"
        mock_sql_file.exists.return_value = True
        mock_path.return_value = mock_sql_file
        
        factory = Factory(xgboost_settings)
        
        # 기존 방식: training_results 없이 PyfuncWrapper 생성
        trained_model = Mock()
        trained_preprocessor = Mock()
        
        wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor
        )
        
        # 기존 속성들 유지 확인
        assert wrapper.trained_model == trained_model
        assert wrapper.trained_preprocessor == trained_preprocessor
        assert hasattr(wrapper, 'loader_sql_snapshot')
        assert hasattr(wrapper, 'augmenter_sql_snapshot')
        
        # 새로운 속성들이 기본값으로 설정됨 확인
        assert wrapper.hyperparameter_optimization["enabled"] is False
        assert wrapper.training_methodology == {}
        assert wrapper.model_class_path == xgboost_settings.model.class_path
    
    def test_new_features_activation(self, xgboost_settings: Settings):
        """새로운 기능들이 올바르게 활성화되는지 테스트"""
        from src.settings import HyperparameterTuningSettings, FeatureStoreSettings
        
        # 새로운 설정들 추가
        xgboost_settings.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=5, metric="accuracy", direction="maximize"
        )
        xgboost_settings.model.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=3, metric="roc_auc", direction="maximize"
        )
        xgboost_settings.feature_store = FeatureStoreSettings(
            provider="dynamic",
            connection_timeout=5000,
            retry_attempts=3,
            connection_info={"redis_host": "localhost:6379"}
        )
        
        factory = Factory(xgboost_settings)
        
        # 새로운 어댑터들 생성 가능 확인
        feature_store_adapter = factory.create_feature_store_adapter()
        optuna_adapter = factory.create_optuna_adapter()
        tuning_utils = factory.create_tuning_utils()
        
        # 올바른 타입인지 확인
        from src.utils.adapters.feature_store_adapter import FeatureStoreAdapter
        from src.utils.adapters.optuna_adapter import OptunaAdapter
        from src.utils.system.tuning_utils import TuningUtils
        
        assert isinstance(feature_store_adapter, FeatureStoreAdapter)
        assert isinstance(optuna_adapter, OptunaAdapter)
        assert isinstance(tuning_utils, TuningUtils)
    
    @patch('src.core.trainer.optuna')
    @patch('src.core.trainer.Factory')
    def test_hyperparameter_optimization_integration(self, mock_factory, mock_optuna, xgboost_settings: Settings):
        """하이퍼파라미터 최적화가 기존 학습과 통합되는지 테스트"""
        from src.settings import HyperparameterTuningSettings
        
        # 최적화 활성화
        xgboost_settings.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=2, metric="accuracy", direction="maximize"
        )
        xgboost_settings.model.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=2, metric="accuracy", direction="maximize"
        )
        
        # Mock 설정
        mock_factory_instance = Mock()
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.1, "n_estimators": 100}
        mock_study.best_value = 0.92
        mock_study.trials = [Mock(), Mock()]
        
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.pruners.MedianPruner.return_value = Mock()
        mock_factory.return_value = mock_factory_instance
        
        trainer = Trainer(xgboost_settings)
        
        # 샘플 데이터
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # objective 함수 Mock
        def mock_optimize(objective, n_trials, timeout):
            # objective 함수를 2번 호출하여 최적화 시뮬레이션
            objective(Mock())
            objective(Mock())
        
        mock_study.optimize = mock_optimize
        
        with patch.object(trainer, '_single_training_iteration') as mock_single:
            mock_single.return_value = {
                'model': Mock(),
                'preprocessor': Mock(),
                'score': 0.92,
                'metrics': {"accuracy": 0.92},
                'training_methodology': {
                    'preprocessing_fit_scope': 'train_only'
                }
            }
            
            # 학습 실행
            result = trainer.train(sample_data, Mock())
            
            # 최적화가 수행되었는지 확인
            mock_optuna.create_study.assert_called_once()
            assert mock_single.call_count >= 1  # objective 함수에서 호출
            
            # 결과에 최적화 메타데이터 포함 확인
            training_results = result[2]
            assert training_results["hyperparameter_optimization"]["enabled"] is True
            assert "best_params" in training_results["hyperparameter_optimization"]
    
    def test_api_backward_compatibility(self):
        """API 응답 스키마의 하위 호환성 테스트"""
        from serving.schemas import PredictionResponse, BatchPredictionResponse
        
        # 기존 필드들로만 응답 생성 가능한지 확인
        old_style_response = PredictionResponse(
            uplift_score=0.85,
            model_uri="runs:/test123/model"
            # optimization_enabled, best_score는 기본값 사용
        )
        
        # 기본값이 올바르게 설정되었는지 확인
        assert old_style_response.uplift_score == 0.85
        assert old_style_response.model_uri == "runs:/test123/model"
        assert old_style_response.optimization_enabled is False
        assert old_style_response.best_score == 0.0
        
        # 새로운 필드들을 포함한 응답도 생성 가능한지 확인
        new_style_response = PredictionResponse(
            uplift_score=0.92,
            model_uri="runs:/optimized456/model",
            optimization_enabled=True,
            best_score=0.92
        )
        
        assert new_style_response.optimization_enabled is True
        assert new_style_response.best_score == 0.92
    
    def test_settings_backward_compatibility(self, xgboost_settings: Settings):
        """Settings 클래스의 하위 호환성 테스트"""
        # 기존 설정들이 모두 유지되는지 확인
        assert hasattr(xgboost_settings, 'environment')
        assert hasattr(xgboost_settings, 'mlflow')
        assert hasattr(xgboost_settings, 'serving')
        assert hasattr(xgboost_settings, 'artifact_stores')
        assert hasattr(xgboost_settings, 'model')
        
        # 새로운 설정들이 Optional로 추가되었는지 확인
        assert hasattr(xgboost_settings, 'hyperparameter_tuning')
        assert hasattr(xgboost_settings, 'feature_store')
        
        # 기본값이 None인지 확인 (하위 호환성)
        assert xgboost_settings.hyperparameter_tuning is None
        assert xgboost_settings.feature_store is None
        
        # 모델 설정도 동일하게 확인
        assert hasattr(xgboost_settings.model, 'hyperparameter_tuning')
        assert xgboost_settings.model.hyperparameter_tuning is None


class TestBlueprintV17GradualActivation:
    """Blueprint v17.0 기능들의 점진적 활성화 테스트"""
    
    def test_feature_store_only_activation(self, xgboost_settings: Settings):
        """Feature Store만 활성화하고 하이퍼파라미터 최적화는 비활성화"""
        from src.settings import FeatureStoreSettings
        
        # Feature Store만 활성화
        xgboost_settings.feature_store = FeatureStoreSettings(
            provider="dynamic",
            connection_timeout=5000,
            retry_attempts=3,
            connection_info={"redis_host": "localhost:6379"}
        )
        
        # 하이퍼파라미터 최적화는 비활성화 유지
        assert xgboost_settings.hyperparameter_tuning is None
        
        factory = Factory(xgboost_settings)
        
        # Feature Store 어댑터는 생성 가능
        feature_store_adapter = factory.create_feature_store_adapter()
        assert feature_store_adapter is not None
        
        # Optuna 어댑터는 생성 불가 (설정 없음)
        with pytest.raises(ValueError, match="Hyperparameter tuning 설정이 없습니다"):
            factory.create_optuna_adapter()
    
    def test_hyperparameter_optimization_only_activation(self, xgboost_settings: Settings):
        """하이퍼파라미터 최적화만 활성화하고 Feature Store는 비활성화"""
        from src.settings import HyperparameterTuningSettings
        
        # 하이퍼파라미터 최적화만 활성화
        xgboost_settings.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=5, metric="accuracy", direction="maximize"
        )
        
        # Feature Store는 비활성화 유지
        assert xgboost_settings.feature_store is None
        
        factory = Factory(xgboost_settings)
        
        # Optuna 어댑터는 생성 가능
        optuna_adapter = factory.create_optuna_adapter()
        assert optuna_adapter is not None
        
        # Feature Store 어댑터는 생성 불가 (설정 없음)
        with pytest.raises(ValueError, match="Feature Store 설정이 없습니다"):
            factory.create_feature_store_adapter()
    
    def test_all_features_activation(self, xgboost_settings: Settings):
        """모든 새로운 기능들을 동시에 활성화"""
        from src.settings import HyperparameterTuningSettings, FeatureStoreSettings
        
        # 모든 새로운 기능 활성화
        xgboost_settings.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=10, metric="accuracy", direction="maximize"
        )
        xgboost_settings.model.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True, n_trials=5, metric="roc_auc", direction="maximize"
        )
        xgboost_settings.feature_store = FeatureStoreSettings(
            provider="dynamic",
            connection_timeout=5000,
            retry_attempts=3,
            connection_info={"redis_host": "localhost:6379"}
        )
        
        factory = Factory(xgboost_settings)
        trainer = Trainer(xgboost_settings)
        
        # 모든 어댑터들 생성 가능 확인
        feature_store_adapter = factory.create_feature_store_adapter()
        optuna_adapter = factory.create_optuna_adapter()
        tuning_utils = factory.create_tuning_utils()
        
        assert feature_store_adapter is not None
        assert optuna_adapter is not None
        assert tuning_utils is not None
        
        # Trainer가 두 조건 모두 확인하는지 테스트
        assert trainer.settings.hyperparameter_tuning.enabled is True
        assert trainer.settings.model.hyperparameter_tuning.enabled is True
        assert trainer.settings.feature_store.provider == "dynamic" 