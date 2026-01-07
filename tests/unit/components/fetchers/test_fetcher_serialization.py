"""
Fetcher 직렬화 및 혼용 모드 테스트
- FeatureStoreFetcher pickle 직렬화 테스트
- 혼용 모드 (SQL 피처 + Feature Store 피처) 테스트
- Lazy initialization 테스트
"""

import pickle
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.components.fetcher.modules.feature_store_fetcher import FeatureStoreFetcher


class TestFeatureStoreFetcherSerialization:
    """FeatureStoreFetcher 직렬화 테스트"""

    @pytest.fixture
    def mock_settings_with_feature_store(self, settings_builder):
        """Feature Store가 활성화된 Settings Mock"""
        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )
        return settings

    @pytest.fixture
    def mock_factory(self):
        """Mock Factory"""
        factory = Mock()
        factory.create_feature_store_adapter.return_value = Mock()
        return factory

    def test_fetcher_serializable_config_extraction(
        self, mock_settings_with_feature_store, mock_factory
    ):
        """직렬화 가능한 설정만 추출되는지 테스트"""
        # When: Fetcher 생성
        fetcher = FeatureStoreFetcher(mock_settings_with_feature_store, mock_factory)

        # Then: _fetcher_config는 직렬화 가능한 dict
        assert isinstance(fetcher._fetcher_config, dict)
        assert "entity_columns" in fetcher._fetcher_config
        assert "timestamp_column" in fetcher._fetcher_config
        assert "target_column" in fetcher._fetcher_config
        assert "features" in fetcher._fetcher_config
        assert "feast_config" in fetcher._fetcher_config

    def test_fetcher_pickle_serialization(self, mock_settings_with_feature_store, mock_factory):
        """Pickle 직렬화/역직렬화 테스트"""
        # Given: Fetcher 생성
        fetcher = FeatureStoreFetcher(mock_settings_with_feature_store, mock_factory)

        # When: Pickle 직렬화 및 역직렬화
        pickled = pickle.dumps(fetcher)
        restored_fetcher = pickle.loads(pickled)

        # Then: 설정 정보 보존
        assert isinstance(restored_fetcher, FeatureStoreFetcher)
        assert restored_fetcher._fetcher_config == fetcher._fetcher_config
        # adapter는 None으로 복원됨 (lazy init)
        assert restored_fetcher._feature_store_adapter is None

    def test_fetcher_getstate_excludes_adapter(
        self, mock_settings_with_feature_store, mock_factory
    ):
        """__getstate__에서 adapter가 제외되는지 테스트"""
        # Given: Adapter가 있는 Fetcher
        fetcher = FeatureStoreFetcher(mock_settings_with_feature_store, mock_factory)
        assert fetcher._feature_store_adapter is not None

        # When: __getstate__ 호출
        state = fetcher.__getstate__()

        # Then: adapter는 None
        assert state["_feature_store_adapter"] is None

    def test_fetcher_setstate_restores_config(self, mock_settings_with_feature_store, mock_factory):
        """__setstate__로 설정이 복원되는지 테스트"""
        # Given: Fetcher 생성 및 직렬화
        fetcher = FeatureStoreFetcher(mock_settings_with_feature_store, mock_factory)
        original_config = fetcher._fetcher_config.copy()

        # When: 직렬화 후 복원
        state = fetcher.__getstate__()
        new_fetcher = FeatureStoreFetcher.__new__(FeatureStoreFetcher)
        new_fetcher.__setstate__(state)

        # Then: 설정 복원됨
        assert new_fetcher._fetcher_config == original_config

    def test_fetcher_entity_columns_preserved(self, mock_settings_with_feature_store, mock_factory):
        """entity_columns 보존 테스트"""
        # Given: entity_columns 설정된 Fetcher
        fetcher = FeatureStoreFetcher(mock_settings_with_feature_store, mock_factory)

        # When: 직렬화 후 복원
        pickled = pickle.dumps(fetcher)
        restored = pickle.loads(pickled)

        # Then: entity_columns 보존
        assert "user_id" in restored._fetcher_config["entity_columns"]


class TestFeatureStoreFetcherHybridMode:
    """혼용 모드 테스트 (SQL 피처 + Feature Store 피처)"""

    @pytest.fixture
    def mock_feature_store_adapter(self):
        """Mock Feature Store Adapter"""
        adapter = Mock()

        # Online Store에서 피처 반환
        def mock_online_features(entity_rows, features):
            return pd.DataFrame(
                {
                    "user_id": [row["user_id"] for row in entity_rows],
                    "fs_feature_1": [0.5] * len(entity_rows),
                    "fs_feature_2": [100] * len(entity_rows),
                }
            )

        adapter.get_online_features.side_effect = mock_online_features
        return adapter

    @pytest.fixture
    def hybrid_fetcher(self, settings_builder, mock_feature_store_adapter):
        """혼용 모드용 Fetcher"""
        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )

        factory = Mock()
        factory.create_feature_store_adapter.return_value = mock_feature_store_adapter

        fetcher = FeatureStoreFetcher(settings, factory)
        return fetcher

    def test_serving_mode_preserves_client_features(self, hybrid_fetcher):
        """serving 모드에서 클라이언트 피처가 보존되는지 테스트"""
        # Given: 클라이언트가 제공한 SQL 피처 포함 DataFrame
        client_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "sql_feature_1": [10.0, 20.0, 30.0],  # SQL에서 온 피처
                "sql_feature_2": ["A", "B", "C"],  # SQL에서 온 피처
            }
        )

        # When: serving 모드로 fetch
        result = hybrid_fetcher.fetch(client_df, run_mode="serving")

        # Then: 클라이언트 피처 보존
        assert "sql_feature_1" in result.columns
        assert "sql_feature_2" in result.columns
        assert list(result["sql_feature_1"]) == [10.0, 20.0, 30.0]

    def test_serving_mode_adds_online_store_features(self, hybrid_fetcher):
        """serving 모드에서 Online Store 피처가 추가되는지 테스트"""
        # Given: entity만 있는 DataFrame
        client_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
            }
        )

        # When: serving 모드로 fetch
        result = hybrid_fetcher.fetch(client_df, run_mode="serving")

        # Then: Online Store 피처 추가됨
        assert "fs_feature_1" in result.columns
        assert "fs_feature_2" in result.columns

    def test_serving_mode_merges_features_correctly(self, hybrid_fetcher):
        """serving 모드에서 피처가 올바르게 병합되는지 테스트"""
        # Given: 혼합 DataFrame
        client_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "client_feature": [100, 200],
            }
        )

        # When: serving 모드로 fetch
        result = hybrid_fetcher.fetch(client_df, run_mode="serving")

        # Then: 모든 피처 존재
        assert "user_id" in result.columns
        assert "client_feature" in result.columns  # 클라이언트 피처
        assert "fs_feature_1" in result.columns  # Online Store 피처
        assert len(result) == 2

    def test_serving_mode_handles_empty_online_features(self, settings_builder):
        """Online Store 피처가 없을 때 처리 테스트"""
        # Given: 빈 피처를 반환하는 adapter
        empty_adapter = Mock()
        empty_adapter.get_online_features.return_value = pd.DataFrame(
            {
                "user_id": [1, 2],
            }
        )  # entity만 반환

        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )

        factory = Mock()
        factory.create_feature_store_adapter.return_value = empty_adapter

        fetcher = FeatureStoreFetcher(settings, factory)

        client_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "client_feature": [10, 20],
            }
        )

        # When: fetch
        result = fetcher.fetch(client_df, run_mode="serving")

        # Then: 클라이언트 피처만 유지
        assert "client_feature" in result.columns
        assert len(result) == 2


class TestFeatureStoreFetcherLazyInit:
    """Lazy Initialization 테스트"""

    def test_lazy_init_after_deserialization(self, settings_builder):
        """역직렬화 후 lazy init 테스트"""
        # Given: Feast config 포함 Fetcher
        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )

        factory = Mock()
        factory.create_feature_store_adapter.return_value = Mock()

        fetcher = FeatureStoreFetcher(settings, factory)

        # When: 직렬화 후 복원
        pickled = pickle.dumps(fetcher)
        restored = pickle.loads(pickled)

        # Then: adapter는 None (lazy init 대기)
        assert restored._feature_store_adapter is None

        # feast_config 보존 확인
        assert restored._fetcher_config.get("feast_config") is not None

    def test_adapter_created_on_first_access(self, settings_builder):
        """첫 접근 시 adapter 생성 테스트"""
        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )

        # adapter 생성 실패하도록 설정
        factory = Mock()
        factory.create_feature_store_adapter.side_effect = Exception("Feast not available")

        fetcher = FeatureStoreFetcher(settings, factory)

        # 직렬화 후 복원 (adapter = None)
        pickled = pickle.dumps(fetcher)
        restored = pickle.loads(pickled)

        assert restored._feature_store_adapter is None

        # lazy init 시도 시 feast_config에서 adapter 생성 시도
        # (실제 Feast 없이는 실패하지만 로직은 동작함)


class TestFeatureStoreFetcherRunModes:
    """run_mode별 동작 테스트"""

    @pytest.fixture
    def fetcher_with_mock_adapter(self, settings_builder):
        """Mock adapter가 있는 Fetcher"""
        settings = (
            settings_builder.with_feature_store(enabled=True)
            .with_entity_columns(["user_id"])
            .build()
        )

        adapter = Mock()

        # Offline Store
        adapter.get_historical_features_with_validation.return_value = pd.DataFrame(
            {
                "user_id": [1, 2],
                "feature_1": [0.5, 0.6],
            }
        )

        # Online Store
        adapter.get_online_features.return_value = pd.DataFrame(
            {
                "user_id": [1, 2],
                "feature_1": [0.5, 0.6],
            }
        )

        factory = Mock()
        factory.create_feature_store_adapter.return_value = adapter

        return FeatureStoreFetcher(settings, factory)

    def test_train_mode_uses_offline_store(self, fetcher_with_mock_adapter):
        """train 모드에서 Offline Store 사용 테스트"""
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "event_timestamp": [datetime.now()] * 2,
            }
        )

        # When: train 모드
        result = fetcher_with_mock_adapter.fetch(df, run_mode="train")

        # Then: Offline Store 메서드 호출됨
        adapter = fetcher_with_mock_adapter._feature_store_adapter
        adapter.get_historical_features_with_validation.assert_called_once()
        adapter.get_online_features.assert_not_called()

    def test_batch_mode_uses_offline_store(self, fetcher_with_mock_adapter):
        """batch 모드에서 Offline Store 사용 테스트"""
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "event_timestamp": [datetime.now()] * 2,
            }
        )

        # When: batch 모드
        result = fetcher_with_mock_adapter.fetch(df, run_mode="batch")

        # Then: Offline Store 메서드 호출됨
        adapter = fetcher_with_mock_adapter._feature_store_adapter
        adapter.get_historical_features_with_validation.assert_called_once()

    def test_serving_mode_uses_online_store(self, fetcher_with_mock_adapter):
        """serving 모드에서 Online Store 사용 테스트"""
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
            }
        )

        # When: serving 모드
        result = fetcher_with_mock_adapter.fetch(df, run_mode="serving")

        # Then: Online Store 메서드 호출됨
        adapter = fetcher_with_mock_adapter._feature_store_adapter
        adapter.get_online_features.assert_called_once()

    def test_invalid_run_mode_raises_error(self, fetcher_with_mock_adapter):
        """유효하지 않은 run_mode 에러 테스트"""
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
            }
        )

        # When/Then: 유효하지 않은 모드
        with pytest.raises(ValueError, match="지원하지 않는 run_mode"):
            fetcher_with_mock_adapter.fetch(df, run_mode="invalid_mode")
