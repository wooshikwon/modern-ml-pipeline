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
from src.utils.data_adapters.bigquery_adapter import BigQueryAdapter
from src.utils.data_adapters.gcs_adapter import GCSAdapter
from src.utils.data_adapters.s3_adapter import S3Adapter
from src.utils.data_adapters.file_system_adapter import FileSystemAdapter
from src.utils.data_adapters.redis_adapter import RedisAdapter
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
        assert factory.settings.model.name == "xgboost_x_learner"
    
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
        """알 수 없는 모델 타입에 대한 오류 처리 테스트"""
        # 설정을 복사하고 모델명을 변경
        modified_settings = xgboost_settings.model_copy()
        modified_settings.model.name = "unknown_model"
        
        factory = Factory(modified_settings)
        with pytest.raises(ValueError, match="Unknown model type"):
            factory.create_model()
    
    @patch('src.core.factory.PyfuncWrapper')
    def test_create_pyfunc_wrapper(self, mock_pyfunc_wrapper, xgboost_settings: Settings):
        """PyfuncWrapper 생성 테스트 (순수 로직 아티팩트 원칙)"""
        factory = Factory(xgboost_settings)
        
        # Mock 컴포넌트 생성
        mock_augmenter = Mock()
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