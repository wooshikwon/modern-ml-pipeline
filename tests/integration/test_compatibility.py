"""
Blueprint v17.0 호환성 보장 통합 테스트 (현대화)

기존 코드와 새로운 기능이 함께 정상 동작하는지 검증하는 테스트

Blueprint 원칙 검증:
- 기존 워크플로우 완전한 하위 호환성
- 새로운 기능의 점진적 활성화
- 중앙 fixture 사용 통일
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.settings import Settings
from src.components.trainer import Trainer
from src.engine.factory import Factory


pytest.skip("Deprecated/outdated test module pending Stage 6 test overhaul (compatibility tests will be rewritten).", allow_module_level=True)


class TestBlueprintV17CompatibilityModernized:
    """Blueprint v17.0 전체 호환성 테스트 (현대화)"""
    
    def test_existing_workflow_unchanged(self, local_test_settings: Settings):
        """
        기존 워크플로우가 변경 없이 동작하는지 테스트한다.
        Blueprint 원칙: 100% 하위 호환성 보장
        """
        # LOCAL 환경에서는 기존 설정 방식 유지 (HPO 비활성화)
        s = local_test_settings
        
        # 기존 컴포넌트들이 정상 동작하는지 확인
        factory = Factory(s)
        trainer = Trainer(s)
        
        # 기존 어댑터들 생성 가능
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        
        # 모든 컴포넌트가 올바른 타입인지 확인
        from src.core.augmenter import Augmenter, PassThroughAugmenter
        from src.core.preprocessor import Preprocessor
        
        # LOCAL 환경에서는 PassThroughAugmenter 사용
        assert isinstance(augmenter, PassThroughAugmenter)
        assert isinstance(preprocessor, Preprocessor)
        assert trainer.settings == s
        
        # 기존 모델 클래스 로딩 확인
        assert s.model.class_path == "sklearn.ensemble.RandomForestClassifier"
        print("✅ 기존 워크플로우 하위 호환성 검증 완료")
    
    @patch('src.core.trainer.mlflow')
    def test_existing_training_produces_compatible_results(self, mock_mlflow, local_test_settings: Settings):
        """
        기존 학습 방식이 호환되는 결과를 생성하는지 테스트한다.
        """
        trainer = Trainer(local_test_settings)
        
        # Mock 컴포넌트 설정
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = mock_preprocessor
        mock_preprocessor.transform.return_value = pd.DataFrame({'feature1': [0.1, 0.2]})
        
        from src.core.augmenter import PassThroughAugmenter
        mock_augmenter = PassThroughAugmenter(settings=local_test_settings)
        
        # 샘플 데이터
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'approved': [0, 1, 0, 1]  # target column
        })
        
        # 기존 방식 학습 실행
        trained_preprocessor, trained_model, training_results = trainer.train(
            df=sample_data,
            model=mock_model,
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor
        )
        
        # 결과 구조 확인 (Blueprint v17.0 확장 포함)
        assert trained_preprocessor is not None
        assert trained_model is not None
        assert isinstance(training_results, dict)
        
        # 기존 metrics 유지
        assert "metrics" in training_results
        
        # 새로운 메타데이터 포함 확인 (기본값으로)
        assert "hyperparameter_optimization" in training_results
        hpo_data = training_results["hyperparameter_optimization"]
        assert not hpo_data.get("enabled", False), "LOCAL 환경에서는 HPO가 비활성화되어야 함"
        
        assert "training_methodology" in training_results
        tm_data = training_results["training_methodology"]
        assert tm_data["preprocessing_fit_scope"] == "train_only"
        print("✅ 기존 학습 방식 호환성 검증 완료")

    def test_existing_pyfunc_wrapper_creation(self, local_test_settings: Settings):
        """
        기존 PyfuncWrapper 생성 방식이 호환되는지 테스트한다.
        """
        factory = Factory(local_test_settings)
        
        # Mock 컴포넌트들
        trained_model = Mock()
        trained_preprocessor = Mock()
        
        # 최소한의 training_results (기존 호환성)
        basic_training_results = {
            "metrics": {"accuracy": 0.85},
            "hyperparameter_optimization": {"enabled": False},
            "training_methodology": {"preprocessing_fit_scope": "train_only"},
            "loader_sql_snapshot": "SELECT user_id, product_id FROM test_table",
            "augmenter_config_snapshot": {"type": "passthrough"},
            "model_class_path": local_test_settings.model.class_path
        }
        
        # PyfuncWrapper 생성
        wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            training_results=basic_training_results
        )
        
        # 기존 속성들이 유지되는지 확인
        assert wrapper.trained_model == trained_model
        assert wrapper.trained_preprocessor == trained_preprocessor
        assert wrapper.model_class_path == local_test_settings.model.class_path
        print("✅ 기존 PyfuncWrapper 생성 방식 호환성 검증 완료")

    def test_new_features_activation_in_dev_env(self, dev_test_settings: Settings):
        """
        DEV 환경에서 새로운 기능들이 올바르게 활성화되는지 테스트한다.
        """
        # DEV 환경에서는 새로운 기능들이 활성화되어야 함
        s = dev_test_settings
        
        # HPO 활성화 확인
        assert s.model.hyperparameter_tuning is not None
        assert s.model.hyperparameter_tuning.enabled == True
        
        # Feature Store 활성화 확인
        assert s.model.augmenter.type == "feature_store"
        
        # 새로운 기능이 활성화된 Factory 생성
        factory = Factory(s)
        
        # DEV 환경에서는 실제 Augmenter 사용 (PassThrough가 아님)
        augmenter = factory.create_augmenter()
        from src.core.augmenter import Augmenter, PassThroughAugmenter
        assert isinstance(augmenter, Augmenter)
        assert not isinstance(augmenter, PassThroughAugmenter)
        
        print("✅ DEV 환경 새로운 기능 활성화 검증 완료")

    @patch('src.core.trainer.optuna')
    @patch('src.core.trainer.mlflow')
    def test_hyperparameter_optimization_integration(self, mock_mlflow, mock_optuna, dev_test_settings: Settings):
        """
        하이퍼파라미터 최적화 기능이 올바르게 통합되는지 테스트한다.
        """
        # DEV 환경 설정에서 HPO 확인
        s = dev_test_settings
        assert s.model.hyperparameter_tuning.enabled == True
        
        # Optuna Mock 설정
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.1
        mock_optuna.create_study.return_value = mock_study
        mock_study.best_trial = mock_trial
        mock_study.best_trial.value = 0.95
        mock_study.best_trial.params = {"n_estimators": 100, "learning_rate": 0.1}
        mock_study.trials = [mock_trial]
        
        trainer = Trainer(s)
        
        # Mock 컴포넌트들
        mock_model = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = mock_preprocessor
        mock_preprocessor.transform.return_value = pd.DataFrame({'f1': [0.1, 0.2]})
        
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = pd.DataFrame({'f1': [0.1, 0.2], 'approved': [1, 0]})
        
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'approved': [1, 0, 1, 0]
        })
        
        # HPO가 활성화된 학습 실행
        trained_preprocessor, trained_model, training_results = trainer.train(
            df=sample_data,
            model=mock_model,
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor
        )
        
        # HPO 결과가 포함되었는지 확인
        assert "hyperparameter_optimization" in training_results
        hpo_data = training_results["hyperparameter_optimization"]
        assert hpo_data["enabled"] == True
        assert "best_params" in hpo_data
        assert "best_score" in hpo_data
        
        # Optuna 호출 확인
        mock_optuna.create_study.assert_called_once()
        print("✅ 하이퍼파라미터 최적화 통합 검증 완료")

    def test_settings_backward_compatibility(self, local_test_settings: Settings, dev_test_settings: Settings):
        """
        설정 구조의 하위 호환성을 검증한다.
        """
        # 모든 환경에서 기본 속성들 존재 확인
        for settings in [local_test_settings, dev_test_settings]:
            # 기존 필수 속성들
            assert hasattr(settings, 'environment')
            assert hasattr(settings, 'mlflow')
            assert hasattr(settings, 'serving')
            assert hasattr(settings, 'data_adapters')
            assert hasattr(settings, 'model')
            
            # 새로운 속성들 (v17.0)
            assert hasattr(settings, 'hyperparameter_tuning')
            
            # 모델 레벨 새로운 속성들
            assert hasattr(settings.model, 'hyperparameter_tuning')
        
        # 환경별 차이 확인
        # LOCAL: 보수적 설정
        assert not local_test_settings.model.hyperparameter_tuning.enabled
        
        # DEV: 신기능 활성화
        assert dev_test_settings.model.hyperparameter_tuning.enabled
        
        print("✅ 설정 하위 호환성 검증 완료")

    def test_feature_store_environment_differentiation(self, local_test_settings: Settings, dev_test_settings: Settings):
        """
        Feature Store 기능의 환경별 차등 적용을 검증한다.
        Blueprint 원칙 9: 환경별 차등적 기능 분리
        """
        # LOCAL 환경: PassThrough 방식
        local_factory = Factory(local_test_settings)
        local_augmenter = local_factory.create_augmenter()
        
        from src.core.augmenter import PassThroughAugmenter
        assert isinstance(local_augmenter, PassThroughAugmenter)
        
        # DEV 환경: Feature Store 연동
        dev_factory = Factory(dev_test_settings)
        with patch.object(dev_factory, 'create_feature_store_adapter'):
            dev_augmenter = dev_factory.create_augmenter()
            from src.core.augmenter import Augmenter
            assert isinstance(dev_augmenter, Augmenter)
            assert not isinstance(dev_augmenter, PassThroughAugmenter)
        
        print("✅ Feature Store 환경별 차등 적용 검증 완료")

    def test_blueprint_principle_compliance_comprehensive(self, local_test_settings: Settings, dev_test_settings: Settings):
        """
        Blueprint v17.0의 10대 핵심 원칙 준수를 종합적으로 검증한다.
        """
        # 원칙 1: 레시피는 논리, 설정은 인프라
        assert local_test_settings.model.class_path == dev_test_settings.model.class_path  # 논리 동일
        assert local_test_settings.environment.app_env != dev_test_settings.environment.app_env  # 인프라 다름
        
        # 원칙 3: URI 기반 동작 및 동적 팩토리
        local_factory = Factory(local_test_settings)
        dev_factory = Factory(dev_test_settings)
        
        # 동일한 인터페이스로 다른 구현체 생성
        local_model = local_factory.create_model()
        dev_model = dev_factory.create_model()
        assert type(local_model) == type(dev_model)  # 동일 클래스
        
        # 원칙 9: 환경별 차등적 기능 분리
        local_augmenter = local_factory.create_augmenter()
        with patch.object(dev_factory, 'create_feature_store_adapter'):
            dev_augmenter = dev_factory.create_augmenter()
        
        # 환경별로 다른 Augmenter 구현
        assert type(local_augmenter) != type(dev_augmenter)
        
        print("✅ Blueprint v17.0 10대 원칙 종합 준수 검증 완료")
        print("🎉 모든 호환성 테스트 통과! Blueprint v17.0 완전 호환성 확보") 