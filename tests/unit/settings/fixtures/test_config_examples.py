"""
Tests for Config Examples - Fixtures System

Phase 1에서 구현된 config_examples.py의 예제 데이터가
유효한 Pydantic 모델로 로드되는지 검증하고 예제 데이터 간 일관성을 확인합니다.
"""

import pytest
from typing import Dict, Any

from tests.fixtures.config_examples import (
    CONFIG_LOCAL_EXAMPLE,
    CONFIG_DEVELOPMENT_EXAMPLE,
    CONFIG_PRODUCTION_EXAMPLE,
    CONFIG_SIMPLE_EXAMPLE,
    CONFIG_EXAMPLES,
    MLFLOW_EXAMPLES,
    DATA_SOURCE_EXAMPLES,
    FEAST_EXAMPLES
)
from src.settings.config import Config


class TestConfigExamplesValidity:
    """Config 예제 데이터가 유효한 Pydantic 모델로 로드되는지 테스트"""

    def test_local_example_loads_as_config(self):
        """Local 예제가 Config 모델로 정상 로드되는지 검증"""
        config = Config(**CONFIG_LOCAL_EXAMPLE)

        assert config.environment.name == "local"
        assert config.mlflow.tracking_uri == "./mlruns"
        assert config.mlflow.experiment_name == "mmp-local"
        assert config.data_source.adapter_type == "sql"
        assert config.feature_store.provider == "feast"
        assert config.serving.enabled is True
        assert config.serving.port == 8000

    def test_development_example_loads_as_config(self):
        """Development 예제가 Config 모델로 정상 로드되는지 검증"""
        config = Config(**CONFIG_DEVELOPMENT_EXAMPLE)

        assert config.environment.name == "development"
        assert "mlflow-dev.company.com" in config.mlflow.tracking_uri
        assert config.data_source.adapter_type == "sql"
        assert config.feature_store.provider == "feast"
        assert config.serving.enabled is True
        assert config.serving.workers == 2

    def test_production_example_loads_as_config(self):
        """Production 예제가 Config 모델로 정상 로드되는지 검증"""
        config = Config(**CONFIG_PRODUCTION_EXAMPLE)

        assert config.environment.name == "production"
        assert "mlflow.company.com" in config.mlflow.tracking_uri
        assert config.data_source.adapter_type == "sql"
        assert config.feature_store.provider == "feast"
        assert config.serving.enabled is True
        assert config.serving.workers == 4

    def test_simple_example_loads_as_config(self):
        """Simple 예제가 Config 모델로 정상 로드되는지 검증"""
        config = Config(**CONFIG_SIMPLE_EXAMPLE)

        assert config.environment.name == "simple"
        assert config.mlflow.tracking_uri == "./mlruns"
        assert config.data_source.adapter_type == "storage"
        assert config.feature_store.provider == "none"

    def test_all_config_examples_load_successfully(self):
        """CONFIG_EXAMPLES 딕셔너리의 모든 예제가 정상 로드되는지 검증"""
        for env_name, example_data in CONFIG_EXAMPLES.items():
            config = Config(**example_data)
            assert config.environment.name == env_name
            assert config.mlflow.tracking_uri is not None
            assert config.mlflow.experiment_name is not None


class TestConfigExamplesConsistency:
    """Config 예제 데이터 간 일관성 검증"""

    def test_mlflow_examples_structure(self):
        """MLflow 예제의 구조 검증"""
        for mlflow_type, mlflow_config in MLFLOW_EXAMPLES.items():
            assert "tracking_uri" in mlflow_config
            assert "experiment_name" in mlflow_config
            assert mlflow_config["tracking_uri"] is not None
            assert mlflow_config["experiment_name"] is not None

    def test_data_source_examples_structure(self):
        """Data Source 예제의 구조 검증"""
        for source_type, source_config in DATA_SOURCE_EXAMPLES.items():
            assert "name" in source_config
            assert "adapter_type" in source_config
            assert "config" in source_config
            assert source_config["adapter_type"] in ["sql", "storage"]

    def test_feast_examples_structure(self):
        """Feast 예제의 구조 검증"""
        for feast_type, feast_config in FEAST_EXAMPLES.items():
            assert "provider" in feast_config

            if feast_config["provider"] == "feast":
                assert "feast_config" in feast_config
                feast_settings = feast_config["feast_config"]
                assert "project" in feast_settings
                assert "registry" in feast_settings
                assert "online_store" in feast_settings
                assert "offline_store" in feast_settings

    def test_environment_progression_consistency(self):
        """환경별 설정의 발전 단계가 일관된지 검증"""
        # Local -> Development -> Production 순서로 복잡도 증가
        local = Config(**CONFIG_LOCAL_EXAMPLE)
        dev = Config(**CONFIG_DEVELOPMENT_EXAMPLE)
        prod = Config(**CONFIG_PRODUCTION_EXAMPLE)

        # Worker 수가 증가하는지 확인
        assert local.serving.workers <= dev.serving.workers <= prod.serving.workers

        # 보안 설정이 향상되는지 확인
        assert not hasattr(local.serving, 'auth') or not getattr(local.serving, 'auth', {}).get('enabled', False)
        # Development와 Production은 인증 활성화
        if hasattr(dev.serving, 'auth'):
            assert dev.serving.auth.enabled is True
        if hasattr(prod.serving, 'auth'):
            assert prod.serving.auth.enabled is True

    def test_feature_store_configuration_consistency(self):
        """Feature Store 설정의 일관성 검증"""
        for env_name, example_data in CONFIG_EXAMPLES.items():
            config = Config(**example_data)

            if config.feature_store.provider == "feast":
                assert hasattr(config.feature_store, 'feast_config')
                assert config.feature_store.feast_config is not None

                feast_config = config.feature_store.feast_config
                assert feast_config.project is not None
                assert feast_config.registry is not None


class TestConfigExamplesSpecificValidation:
    """특정 Config 예제에 대한 상세한 검증"""

    def test_local_config_minimal_setup(self):
        """Local 설정이 최소한의 구성인지 확인"""
        config = Config(**CONFIG_LOCAL_EXAMPLE)

        # Local은 파일 기반으로 설정
        assert config.mlflow.tracking_uri.startswith("./")
        assert config.feature_store.feast_config.registry.endswith(".db")
        assert config.feature_store.feast_config.online_store["type"] == "sqlite"
        assert config.feature_store.feast_config.offline_store["type"] == "file"

    def test_development_config_cloud_integration(self):
        """Development 설정이 클라우드 통합을 포함하는지 확인"""
        config = Config(**CONFIG_DEVELOPMENT_EXAMPLE)

        # HTTP 기반 MLflow
        assert config.mlflow.tracking_uri.startswith("http")

        # 클라우드 스토리지 사용
        if hasattr(config, 'artifact_store'):
            assert config.artifact_store.type == "s3"

        # Redis 사용
        assert config.feature_store.feast_config.online_store["type"] == "redis"
        assert config.feature_store.feast_config.offline_store["type"] == "bigquery"

    def test_production_config_security_features(self):
        """Production 설정이 보안 기능을 포함하는지 확인"""
        config = Config(**CONFIG_PRODUCTION_EXAMPLE)

        # HTTPS 사용
        assert config.mlflow.tracking_uri.startswith("https")

        # 강화된 온라인 스토어 (DynamoDB)
        assert config.feature_store.feast_config.online_store["type"] == "dynamodb"

        # S3 암호화 설정
        if hasattr(config, 'artifact_store'):
            if hasattr(config.artifact_store.config, 'kms_key_id'):
                assert config.artifact_store.config.kms_key_id is not None

    def test_sql_adapter_configurations(self):
        """SQL 어댑터 설정의 다양성 검증"""
        postgresql_config = DATA_SOURCE_EXAMPLES["postgresql"]
        mysql_config = DATA_SOURCE_EXAMPLES["mysql"]

        # 다른 데이터베이스 URI 형태
        assert "postgresql://" in postgresql_config["config"]["connection_uri"]
        assert "mysql://" in mysql_config["config"]["connection_uri"]

        # 공통 설정 존재
        assert "query_timeout" in postgresql_config["config"]
        assert "query_timeout" in mysql_config["config"]

    def test_storage_adapter_configurations(self):
        """Storage 어댑터 설정의 다양성 검증"""
        local_storage = DATA_SOURCE_EXAMPLES["local_storage"]
        s3_storage = DATA_SOURCE_EXAMPLES["s3_storage"]

        # 로컬 vs 클라우드 경로
        assert local_storage["config"]["base_path"].startswith("./")
        assert s3_storage["config"]["base_path"].startswith("s3://")

        # 지원 형식 확인
        assert "supported_formats" in local_storage["config"]
        assert "csv" in local_storage["config"]["supported_formats"]

    def test_output_configurations_optional(self):
        """Output 설정이 선택사항임을 확인"""
        # Simple config는 output 설정이 없음
        simple_config = Config(**CONFIG_SIMPLE_EXAMPLE)
        assert not hasattr(simple_config, 'output') or simple_config.output is None

        # Development/Production은 output 설정 있음
        dev_config = Config(**CONFIG_DEVELOPMENT_EXAMPLE)
        if hasattr(dev_config, 'output') and dev_config.output:
            assert hasattr(dev_config.output, 'inference')
            assert hasattr(dev_config.output, 'preprocessed')


class TestConfigExamplesEdgeCases:
    """Config 예제의 경계 사례 및 특수 케이스 테스트"""

    def test_feature_store_disabled_configuration(self):
        """Feature Store 비활성화 설정 검증"""
        # Simple config에서 feature store 비활성화
        config = Config(**CONFIG_SIMPLE_EXAMPLE)
        assert config.feature_store.provider == "none"

        # 비활성화시 feast_config가 없어도 됨
        assert not hasattr(config.feature_store, 'feast_config') or \
               config.feature_store.feast_config is None

    def test_serving_optional_configurations(self):
        """Serving의 선택적 설정들 검증"""
        configs = [
            Config(**CONFIG_LOCAL_EXAMPLE),
            Config(**CONFIG_DEVELOPMENT_EXAMPLE),
            Config(**CONFIG_PRODUCTION_EXAMPLE)
        ]

        for config in configs:
            # 기본 설정 존재
            assert config.serving.enabled is not None
            assert config.serving.port is not None

            # 선택적 설정들
            if hasattr(config.serving, 'host'):
                assert isinstance(config.serving.host, str)
            if hasattr(config.serving, 'workers'):
                assert config.serving.workers >= 1

    def test_mlflow_authentication_optional(self):
        """MLflow 인증 설정이 선택사항임을 확인"""
        local_config = Config(**CONFIG_LOCAL_EXAMPLE)

        # Local은 인증 없음
        assert local_config.mlflow.tracking_username == ""
        assert local_config.mlflow.tracking_password == ""

        # Development/Production은 인증 있을 수 있음
        dev_config = Config(**CONFIG_DEVELOPMENT_EXAMPLE)
        if dev_config.mlflow.tracking_username:
            assert len(dev_config.mlflow.tracking_username) > 0
        if dev_config.mlflow.tracking_password:
            assert len(dev_config.mlflow.tracking_password) > 0

    def test_adapter_config_flexibility(self):
        """어댑터 config 필드의 유연성 검증"""
        for source_type, source_config in DATA_SOURCE_EXAMPLES.items():
            config_dict = source_config["config"]

            # config는 딕셔너리이고 다양한 필드를 가질 수 있음
            assert isinstance(config_dict, dict)

            # SQL 어댑터는 connection_uri 필수
            if source_config["adapter_type"] == "sql":
                assert "connection_uri" in config_dict or \
                       "project_id" in config_dict  # BigQuery style

            # Storage 어댑터는 base_path 필수
            elif source_config["adapter_type"] == "storage":
                assert "base_path" in config_dict

    def test_environment_name_validation(self):
        """환경 이름이 유효한지 검증"""
        valid_environments = ["local", "development", "production", "simple"]

        for env_name in valid_environments:
            assert env_name in CONFIG_EXAMPLES
            config = Config(**CONFIG_EXAMPLES[env_name])
            assert config.environment.name == env_name
            assert len(config.environment.name) > 0

    def test_feast_online_offline_store_types(self):
        """Feast의 다양한 Online/Offline Store 타입 검증"""
        feast_examples = [
            FEAST_EXAMPLES["local_sqlite"],
            FEAST_EXAMPLES["cloud_setup"]
        ]

        for feast_config in feast_examples:
            if feast_config["provider"] == "feast":
                feast_settings = feast_config["feast_config"]
                online_store = feast_settings["online_store"]
                offline_store = feast_settings["offline_store"]

                # Online Store 타입 검증
                assert online_store["type"] in ["sqlite", "redis", "dynamodb"]

                # Offline Store 타입 검증
                assert offline_store["type"] in ["file", "bigquery"]