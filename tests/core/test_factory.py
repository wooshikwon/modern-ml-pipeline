"""
Factory 컴포넌트 테스트

Blueprint 원칙 검증:
- URI 기반 동작 및 동적 팩토리 원칙
- 통합 데이터 어댑터 원칙
- 순수 로직 아티팩트 원칙
"""

import pytest
from unittest.mock import Mock, patch
from src.core.factory import Factory
from src.settings.settings import Settings
from src.utils.adapters.bigquery_adapter import BigQueryAdapter
from src.utils.adapters.gcs_adapter import GCSAdapter
from src.utils.adapters.s3_adapter import S3Adapter
from src.utils.adapters.file_system_adapter import FileSystemAdapter
from src.utils.adapters.redis_adapter import RedisAdapter
from src.core.augmenter import Augmenter
from src.core.preprocessor import Preprocessor
from src.core.trainer import Trainer
from src.models.xgboost_x_learner import XGBoostXLearner
from src.models.causal_forest import CausalForestModel


class TestFactory:
    """Factory 컴포넌트 테스트"""
    
    def test_factory_initialization(self, xgboost_settings: Settings):
        """Factory가 올바른 설정으로 초기화되는지 테스트"""
        factory = Factory(xgboost_settings)
        assert factory.settings == xgboost_settings
        assert factory.settings.model.class_path == "src.models.xgboost_x_learner.XGBoostXLearner"
    
    def test_create_data_adapter_bigquery(self, xgboost_settings: Settings):
        """BigQuery 어댑터 생성 테스트"""
        factory = Factory(xgboost_settings)
        adapter = factory.create_data_adapter("bq")
        assert isinstance(adapter, BigQueryAdapter)
        assert adapter.settings == xgboost_settings
    
    def test_create_data_adapter_gcs(self, xgboost_settings: Settings):
        """GCS 어댑터 생성 테스트"""
        factory = Factory(xgboost_settings)
        adapter = factory.create_data_adapter("gs")
        assert isinstance(adapter, GCSAdapter)
        assert adapter.settings == xgboost_settings
    
    def test_create_data_adapter_s3(self, xgboost_settings: Settings):
        """S3 어댑터 생성 테스트"""
        factory = Factory(xgboost_settings)
        adapter = factory.create_data_adapter("s3")
        assert isinstance(adapter, S3Adapter)
        assert adapter.settings == xgboost_settings
    
    def test_create_data_adapter_file(self, xgboost_settings: Settings):
        """FileSystem 어댑터 생성 테스트"""
        factory = Factory(xgboost_settings)
        adapter = factory.create_data_adapter("file")
        assert isinstance(adapter, FileSystemAdapter)
        assert adapter.settings == xgboost_settings
    
    def test_create_data_adapter_redis(self, xgboost_settings: Settings):
        """Redis 어댑터 생성 테스트 (선택적 의존성)"""
        factory = Factory(xgboost_settings)
        try:
            adapter = factory.create_data_adapter("redis")
            assert isinstance(adapter, RedisAdapter)
        except ImportError:
            # Redis가 설치되지 않은 경우 적절한 오류 발생
            pytest.skip("Redis not available")
    
    def test_create_data_adapter_unknown_scheme(self, xgboost_settings: Settings):
        """알 수 없는 스킴에 대한 오류 처리 테스트"""
        factory = Factory(xgboost_settings)
        with pytest.raises(ValueError, match="Unknown data adapter scheme"):
            factory.create_data_adapter("unknown")
    
    def test_create_augmenter(self, xgboost_settings: Settings):
        """Augmenter 생성 테스트"""
        factory = Factory(xgboost_settings)
        augmenter = factory.create_augmenter()
        assert isinstance(augmenter, Augmenter)
        assert augmenter.settings == xgboost_settings
    
    def test_create_preprocessor(self, xgboost_settings: Settings):
        """Preprocessor 생성 테스트"""
        factory = Factory(xgboost_settings)
        preprocessor = factory.create_preprocessor()
        assert isinstance(preprocessor, Preprocessor)
        assert preprocessor.settings == xgboost_settings
    
    def test_create_trainer(self, xgboost_settings: Settings):
        """Trainer 생성 테스트"""
        factory = Factory(xgboost_settings)
        trainer = factory.create_trainer()
        assert isinstance(trainer, Trainer)
        assert trainer.settings == xgboost_settings
    
    def test_create_model_xgboost(self, xgboost_settings: Settings):
        """XGBoost 모델 생성 테스트"""
        factory = Factory(xgboost_settings)
        model = factory.create_model()
        assert isinstance(model, XGBoostXLearner)
        assert model.settings == xgboost_settings
    
    def test_create_model_causal_forest(self, causal_forest_settings: Settings):
        """CausalForest 모델 생성 테스트"""
        factory = Factory(causal_forest_settings)
        model = factory.create_model()
        assert isinstance(model, CausalForestModel)
        assert model.settings == causal_forest_settings
    
    def test_create_model_unknown_type(self, xgboost_settings: Settings):
        """잘못된 class_path에 대한 오류 처리 테스트 (동적 모델 로딩)"""
        # 설정을 복사하고 class_path를 변경
        modified_settings = xgboost_settings.model_copy()
        modified_settings.model.class_path = "invalid.module.path.UnknownModel"
        
        factory = Factory(modified_settings)
        with pytest.raises(ValueError, match="모델 클래스를 로드할 수 없습니다"):
            factory.create_model()
    
    def test_dynamic_model_loading_external_model(self, xgboost_settings: Settings):
        """외부 모델 동적 로딩 테스트 (Blueprint v13.0 핵심 기능)"""
        # 설정을 복사하고 외부 모델 class_path로 변경 (예: scikit-learn)
        modified_settings = xgboost_settings.model_copy()
        modified_settings.model.class_path = "sklearn.ensemble.RandomForestRegressor"
        modified_settings.model.hyperparameters.root = {"n_estimators": 100, "random_state": 42}
        
        factory = Factory(modified_settings)
        model = factory.create_model()
        
        # 동적으로 로드된 모델이 올바른 타입인지 확인
        assert model.__class__.__name__ == "RandomForestRegressor"
        assert hasattr(model, "fit")  # scikit-learn 인터페이스 확인
        assert hasattr(model, "predict")
    
    def test_create_complete_wrapped_artifact(self, xgboost_settings: Settings):
        """완전한 Wrapped Artifact 생성 테스트 (Blueprint v13.0)"""
        factory = Factory(xgboost_settings)
        
        # Mock 학습된 컴포넌트들
        mock_trained_model = Mock()
        mock_trained_preprocessor = Mock()
        
        with patch.object(factory, '_create_loader_sql_snapshot', return_value="SELECT * FROM test_table"):
            with patch.object(factory, '_create_augmenter_sql_snapshot', return_value="SELECT feature1 FROM features"):
                with patch.object(factory, '_create_recipe_yaml_snapshot', return_value="model:\n  class_path: test"):
                    with patch.object(factory, '_create_training_metadata') as mock_metadata:
                        mock_metadata.return_value = {
                            "training_timestamp": "2024-01-01T00:00:00",
                            "model_class": "XGBoostXLearner",
                            "recipe_file": "test_recipe",
                            "run_name": "XGBoostXLearner_test_recipe_20240101_000000"
                        }
                        
                        wrapper = factory.create_pyfunc_wrapper(mock_trained_model, mock_trained_preprocessor)
                        
                        # 완전한 Wrapped Artifact 검증
                        assert wrapper.trained_model == mock_trained_model
                        assert wrapper.trained_preprocessor == mock_trained_preprocessor
                        assert wrapper.loader_sql_snapshot == "SELECT * FROM test_table"
                        assert wrapper.augmenter_sql_snapshot == "SELECT feature1 FROM features"
                        assert wrapper.recipe_yaml_snapshot == "model:\n  class_path: test"
                        assert "training_timestamp" in wrapper.training_metadata
                        
                        # 하위 호환성 별칭 검증
                        assert wrapper.model == mock_trained_model
                        assert wrapper.preprocessor == mock_trained_preprocessor
        mock_preprocessor = Mock()
        mock_model = Mock()
        
        # PyfuncWrapper 생성
        wrapper = factory.create_pyfunc_wrapper(
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor,
            model=mock_model
        )
        
        # PyfuncWrapper 생성자가 올바른 인자로 호출되었는지 확인
        mock_pyfunc_wrapper.assert_called_once_with(
            augmenter=mock_augmenter,
            preprocessor=mock_preprocessor,
            model=mock_model,
            settings=xgboost_settings
        )
    
    def test_blueprint_principle_uri_driven_operation(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: URI 기반 동작"""
        factory = Factory(xgboost_settings)
        
        # URI 스킴별 어댑터 생성이 올바르게 동작하는지 확인
        uri_scheme_mapping = {
            "bq": BigQueryAdapter,
            "gs": GCSAdapter,
            "s3": S3Adapter,
            "file": FileSystemAdapter,
        }
        
        for scheme, expected_adapter_class in uri_scheme_mapping.items():
            adapter = factory.create_data_adapter(scheme)
            assert isinstance(adapter, expected_adapter_class)
    
    def test_blueprint_principle_unified_data_adapter(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: 통합 데이터 어댑터"""
        factory = Factory(xgboost_settings)
        
        # 모든 어댑터가 BaseDataAdapter를 상속받는지 확인
        from src.interface.base_data_adapter import BaseDataAdapter
        
        for scheme in ["bq", "gs", "s3", "file"]:
            adapter = factory.create_data_adapter(scheme)
            assert isinstance(adapter, BaseDataAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
    
    def test_blueprint_principle_pure_logic_artifact(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: 순수 로직 아티팩트"""
        factory = Factory(xgboost_settings)
        
        # 생성되는 모든 컴포넌트가 설정을 받지만 인프라 정보를 직접 포함하지 않는지 확인
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        
        # 컴포넌트들이 설정을 참조하지만 하드코딩된 인프라 정보를 포함하지 않는지 확인
        assert augmenter.settings == xgboost_settings
        assert preprocessor.settings == xgboost_settings
        assert model.settings == xgboost_settings
        
        # 인프라 정보는 설정을 통해서만 접근 가능해야 함
        assert hasattr(xgboost_settings, 'data_sources')
        assert hasattr(xgboost_settings, 'mlflow')


# 🆕 Blueprint v17.0: 새로운 어댑터 및 확장 기능 테스트 클래스
class TestFactoryBlueprintV17Extensions:
    """Blueprint v17.0에서 추가된 새로운 어댑터들과 확장 기능 테스트"""
    
    def test_create_feature_store_adapter(self, xgboost_settings: Settings):
        """FeatureStoreAdapter 생성 테스트"""
        from src.settings.settings import FeatureStoreSettings
        
        # FeatureStore 설정 추가
        xgboost_settings.feature_store = FeatureStoreSettings(
            provider="dynamic",
            connection_timeout=5000,
            retry_attempts=3,
            connection_info={"redis_host": "localhost:6379"}
        )
        
        factory = Factory(xgboost_settings)
        
        # FeatureStoreAdapter 생성
        adapter = factory.create_feature_store_adapter()
        
        # 올바른 타입인지 확인
        from src.utils.adapters.feature_store_adapter import FeatureStoreAdapter
        assert isinstance(adapter, FeatureStoreAdapter)
        assert adapter.settings == xgboost_settings
        assert adapter.feature_store_config == xgboost_settings.feature_store
    
    def test_create_feature_store_adapter_without_settings(self, xgboost_settings: Settings):
        """FeatureStore 설정 없이 어댑터 생성 시 오류 테스트"""
        # feature_store 설정을 None으로 설정
        xgboost_settings.feature_store = None
        
        factory = Factory(xgboost_settings)
        
        # ValueError 발생 확인
        with pytest.raises(ValueError, match="Feature Store 설정이 없습니다"):
            factory.create_feature_store_adapter()
    
    def test_create_optuna_adapter(self, xgboost_settings: Settings):
        """OptunaAdapter 생성 테스트"""
        from src.settings.settings import HyperparameterTuningSettings
        
        # 하이퍼파라미터 튜닝 설정 추가
        xgboost_settings.hyperparameter_tuning = HyperparameterTuningSettings(
            enabled=True,
            n_trials=10,
            metric="accuracy",
            direction="maximize"
        )
        
        factory = Factory(xgboost_settings)
        
        # OptunaAdapter 생성
        adapter = factory.create_optuna_adapter()
        
        # 올바른 타입인지 확인
        from src.utils.adapters.optuna_adapter import OptunaAdapter
        assert isinstance(adapter, OptunaAdapter)
        assert adapter.settings == xgboost_settings.hyperparameter_tuning
    
    def test_create_optuna_adapter_without_settings(self, xgboost_settings: Settings):
        """하이퍼파라미터 튜닝 설정 없이 OptunaAdapter 생성 시 오류 테스트"""
        # hyperparameter_tuning 설정을 None으로 설정
        xgboost_settings.hyperparameter_tuning = None
        
        factory = Factory(xgboost_settings)
        
        # ValueError 발생 확인
        with pytest.raises(ValueError, match="Hyperparameter tuning 설정이 없습니다"):
            factory.create_optuna_adapter()
    
    def test_create_tuning_utils(self, xgboost_settings: Settings):
        """TuningUtils 생성 테스트"""
        factory = Factory(xgboost_settings)
        
        # TuningUtils 생성
        utils = factory.create_tuning_utils()
        
        # 올바른 타입인지 확인
        from src.utils.system.tuning_utils import TuningUtils
        assert isinstance(utils, TuningUtils)
    
    @patch('src.core.factory.Path')
    def test_create_pyfunc_wrapper_with_training_results(self, mock_path, xgboost_settings: Settings):
        """확장된 PyfuncWrapper 생성 테스트 (training_results 포함)"""
        # Mock 설정
        mock_sql_file = Mock()
        mock_sql_file.read_text.return_value = "SELECT user_id, feature1 FROM table"
        mock_sql_file.exists.return_value = True
        mock_path.return_value = mock_sql_file
        
        factory = Factory(xgboost_settings)
        
        # Mock 컴포넌트들
        trained_model = Mock()
        trained_preprocessor = Mock()
        
        # 🆕 training_results 포함
        training_results = {
            "metrics": {"accuracy": 0.92},
            "hyperparameter_optimization": {
                "enabled": True,
                "best_params": {"learning_rate": 0.1, "n_estimators": 100},
                "best_score": 0.92,
                "total_trials": 50
            },
            "training_methodology": {
                "train_test_split_method": "stratified",
                "preprocessing_fit_scope": "train_only",
                "random_state": 42
            }
        }
        
        # 확장된 PyfuncWrapper 생성
        wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            training_results=training_results
        )
        
        # 확장된 속성들 확인
        assert wrapper.model_class_path == xgboost_settings.model.class_path
        assert wrapper.hyperparameter_optimization["enabled"] is True
        assert wrapper.hyperparameter_optimization["best_params"]["learning_rate"] == 0.1
        assert wrapper.training_methodology["preprocessing_fit_scope"] == "train_only"
        
        # 기존 속성들도 유지되는지 확인
        assert wrapper.trained_model == trained_model
        assert wrapper.trained_preprocessor == trained_preprocessor
    
    @patch('src.core.factory.Path')
    def test_create_pyfunc_wrapper_backward_compatibility(self, mock_path, xgboost_settings: Settings):
        """PyfuncWrapper 하위 호환성 테스트 (training_results 없이)"""
        # Mock 설정
        mock_sql_file = Mock()
        mock_sql_file.read_text.return_value = "SELECT user_id, feature1 FROM table"
        mock_sql_file.exists.return_value = True
        mock_path.return_value = mock_sql_file
        
        factory = Factory(xgboost_settings)
        
        # Mock 컴포넌트들
        trained_model = Mock()
        trained_preprocessor = Mock()
        
        # 기존 방식으로 PyfuncWrapper 생성 (training_results 없이)
        wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor
        )
        
        # 기본값들이 올바르게 설정되었는지 확인
        assert wrapper.model_class_path == xgboost_settings.model.class_path
        assert wrapper.hyperparameter_optimization["enabled"] is False
        assert wrapper.training_methodology == {}
        
        # 기존 속성들이 유지되는지 확인
        assert wrapper.trained_model == trained_model
        assert wrapper.trained_preprocessor == trained_preprocessor
    
    def test_enhanced_pyfunc_wrapper_predict_metadata(self, xgboost_settings: Settings):
        """확장된 PyfuncWrapper의 predict 메서드 메타데이터 포함 테스트"""
        from src.core.factory import PyfuncWrapper
        
        # Mock 컴포넌트들
        trained_model = Mock()
        trained_preprocessor = Mock()
        trained_augmenter = Mock()
        
        # 최적화 결과 포함
        hyperparameter_optimization = {
            "enabled": True,
            "best_params": {"learning_rate": 0.1},
            "best_score": 0.92
        }
        
        training_methodology = {
            "preprocessing_fit_scope": "train_only",
            "train_test_split_method": "stratified"
        }
        
        # 확장된 PyfuncWrapper 생성
        wrapper = PyfuncWrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_augmenter=trained_augmenter,
            loader_sql_snapshot="SELECT user_id FROM table",
            augmenter_sql_snapshot="SELECT * FROM features",
            recipe_yaml_snapshot="model: test",
            training_metadata={},
            model_class_path="test.Model",
            hyperparameter_optimization=hyperparameter_optimization,
            training_methodology=training_methodology
        )
        
        # Mock 예측 설정
        input_df = pd.DataFrame({"user_id": [1, 2, 3]})
        predictions_df = pd.DataFrame({"user_id": [1, 2, 3], "uplift_score": [0.1, 0.2, 0.3]})
        
        trained_augmenter.augment_batch.return_value = input_df
        trained_model.predict.return_value = predictions_df["uplift_score"].values
        
        # return_intermediate=True로 예측 실행
        result = wrapper.predict(None, input_df, params={"run_mode": "batch", "return_intermediate": True})
        
        # 메타데이터가 포함되었는지 확인
        assert "hyperparameter_optimization" in result
        assert "training_methodology" in result
        assert result["hyperparameter_optimization"]["enabled"] is True
        assert result["training_methodology"]["preprocessing_fit_scope"] == "train_only" 