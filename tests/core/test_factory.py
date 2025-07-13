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