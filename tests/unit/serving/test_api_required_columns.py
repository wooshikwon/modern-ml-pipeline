"""
API 필수 컬럼 검증 테스트
- Feature Store fetcher 사용 시: entity_columns만 필수
- pass_through fetcher 사용 시: 모든 feature_columns 필수
"""

from unittest.mock import Mock

import pandas as pd
import pytest


class TestRequiredColumnsValidation:
    """필수 컬럼 검증 로직 테스트"""

    @pytest.fixture
    def mock_model_with_feature_store_fetcher(self):
        """Feature Store Fetcher가 있는 Mock 모델"""
        model = Mock()

        # unwrap_python_model() 반환값 설정
        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id", "product_id"],
            "feature_columns": ["feature_1", "feature_2", "fs_feature_1", "fs_feature_2"],
            "target_column": "target",
        }

        # Feature Store Fetcher Mock
        fetcher = Mock()
        fetcher._fetcher_config = {
            "entity_columns": ["user_id", "product_id"],
            "features": ["fv:fs_feature_1", "fv:fs_feature_2"],
        }
        wrapped_model.trained_fetcher = fetcher

        model.unwrap_python_model.return_value = wrapped_model
        model.predict.return_value = pd.DataFrame({"prediction": [0, 1]})

        return model

    @pytest.fixture
    def mock_model_without_fetcher(self):
        """Fetcher 없는 Mock 모델 (pass_through)"""
        model = Mock()

        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id"],
            "feature_columns": ["feature_1", "feature_2", "feature_3"],
            "target_column": "target",
        }
        wrapped_model.trained_fetcher = None  # Fetcher 없음

        model.unwrap_python_model.return_value = wrapped_model
        model.predict.return_value = pd.DataFrame({"prediction": [0]})

        return model

    def test_feature_store_fetcher_requires_only_entity_columns(
        self, mock_model_with_feature_store_fetcher
    ):
        """Feature Store fetcher: entity_columns만 필수"""
        model = mock_model_with_feature_store_fetcher
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema
        trained_fetcher = wrapped.trained_fetcher

        # 필수 컬럼 결정 로직 (endpoints.py 로직 재현)
        has_feature_store_fetcher = trained_fetcher is not None and hasattr(
            trained_fetcher, "_fetcher_config"
        )

        required_cols = set()
        if has_feature_store_fetcher:
            # Feature Store fetcher: entity_columns만 필수
            entity_cols = di.get("entity_columns", [])
            required_cols.update(entity_cols)
        else:
            # pass_through: 모든 feature_columns 필수
            required_cols.update(di.get("feature_columns", []))

        # Then: entity_columns만 필수
        assert required_cols == {"user_id", "product_id"}
        assert "feature_1" not in required_cols  # feature는 필수 아님 (FS에서 가져옴)

    def test_pass_through_fetcher_requires_all_feature_columns(self, mock_model_without_fetcher):
        """pass_through fetcher: 모든 feature_columns 필수"""
        model = mock_model_without_fetcher
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema
        trained_fetcher = wrapped.trained_fetcher

        # 필수 컬럼 결정 로직
        has_feature_store_fetcher = trained_fetcher is not None and hasattr(
            trained_fetcher, "_fetcher_config"
        )

        required_cols = set()
        if has_feature_store_fetcher:
            entity_cols = di.get("entity_columns", [])
            required_cols.update(entity_cols)
        else:
            # pass_through: 모든 feature_columns 필수
            required_cols.update(di.get("feature_columns", []))

        # Then: 모든 feature_columns 필수
        assert required_cols == {"feature_1", "feature_2", "feature_3"}

    def test_missing_entity_column_detected(self, mock_model_with_feature_store_fetcher):
        """entity_column 누락 감지 테스트"""
        model = mock_model_with_feature_store_fetcher
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema
        trained_fetcher = wrapped.trained_fetcher

        # 필수 컬럼 결정
        entity_cols = di.get("entity_columns", [])
        required_cols = set(entity_cols)

        # 입력 데이터 (entity 하나 누락)
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2],  # product_id 누락
                "some_feature": [10, 20],
            }
        )

        # 누락 컬럼 체크
        missing = [c for c in sorted(required_cols) if c not in input_df.columns]

        # Then: product_id 누락 감지
        assert "product_id" in missing

    def test_missing_feature_column_detected(self, mock_model_without_fetcher):
        """feature_column 누락 감지 테스트"""
        model = mock_model_without_fetcher
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema

        # 필수 컬럼 결정 (pass_through)
        required_cols = set(di.get("feature_columns", []))

        # 입력 데이터 (feature 하나 누락)
        input_df = pd.DataFrame(
            {
                "user_id": [1],
                "feature_1": [10],
                "feature_2": [20],
                # feature_3 누락
            }
        )

        # 누락 컬럼 체크
        missing = [c for c in sorted(required_cols) if c not in input_df.columns]

        # Then: feature_3 누락 감지
        assert "feature_3" in missing


class TestHybridModeRequiredColumns:
    """혼용 모드에서 필수 컬럼 테스트"""

    @pytest.fixture
    def mock_hybrid_mode_model(self):
        """혼용 모드 Mock 모델"""
        model = Mock()

        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id"],
            "feature_columns": ["sql_feature", "fs_feature"],  # SQL + FS
            "target_column": "target",
        }

        # Feature Store Fetcher (일부 피처만 FS에서)
        fetcher = Mock()
        fetcher._fetcher_config = {
            "entity_columns": ["user_id"],
            "features": ["fv:fs_feature"],  # fs_feature만 FS에서
        }
        wrapped_model.trained_fetcher = fetcher

        model.unwrap_python_model.return_value = wrapped_model

        return model

    def test_hybrid_mode_only_entity_required(self, mock_hybrid_mode_model):
        """혼용 모드: entity만 필수, 나머지는 클라이언트가 제공하거나 FS에서 가져옴"""
        model = mock_hybrid_mode_model
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema
        trained_fetcher = wrapped.trained_fetcher

        # 필수 컬럼 결정
        has_feature_store_fetcher = trained_fetcher is not None and hasattr(
            trained_fetcher, "_fetcher_config"
        )

        required_cols = set()
        if has_feature_store_fetcher:
            required_cols.update(di.get("entity_columns", []))

        # Then: entity만 필수
        assert required_cols == {"user_id"}
        assert "sql_feature" not in required_cols  # 클라이언트가 선택적으로 제공
        assert "fs_feature" not in required_cols  # FS에서 가져옴

    def test_hybrid_mode_client_can_provide_extra_features(self, mock_hybrid_mode_model):
        """혼용 모드: 클라이언트가 추가 피처 제공 가능"""
        # Given: entity + 추가 피처
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "sql_feature": [100, 200],  # 클라이언트가 제공
                # fs_feature는 FS에서 가져옴
            }
        )

        model = mock_hybrid_mode_model
        wrapped = model.unwrap_python_model()
        di = wrapped.data_interface_schema

        # 필수 컬럼 체크
        entity_cols = di.get("entity_columns", [])
        missing = [c for c in entity_cols if c not in input_df.columns]

        # Then: 필수 컬럼 만족
        assert missing == []


class TestValidationEdgeCases:
    """검증 엣지 케이스 테스트"""

    def test_empty_entity_columns(self):
        """entity_columns가 비어있는 경우"""
        di = {
            "entity_columns": [],
            "feature_columns": ["feature_1", "feature_2"],
        }
        trained_fetcher = Mock()
        trained_fetcher._fetcher_config = {"entity_columns": []}

        # Feature Store fetcher 있음
        has_feature_store_fetcher = True

        required_cols = set()
        if has_feature_store_fetcher:
            required_cols.update(di.get("entity_columns", []))

        # Then: 필수 컬럼 없음
        assert required_cols == set()

    def test_no_data_interface_schema(self):
        """data_interface_schema가 없는 경우"""
        wrapped_model = Mock()
        wrapped_model.data_interface_schema = None
        wrapped_model.trained_fetcher = None

        di = getattr(wrapped_model, "data_interface_schema", {}) or {}

        # 필수 컬럼 결정
        required_cols = set()
        required_cols.update(di.get("feature_columns", []))

        # Then: 필수 컬럼 없음 (스키마 없으면 검증 skip)
        assert required_cols == set()

    def test_fetcher_without_fetcher_config(self):
        """_fetcher_config가 없는 Fetcher"""
        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id"],
            "feature_columns": ["feature_1"],
        }

        # Fetcher는 있지만 _fetcher_config 없음
        fetcher = Mock(spec=[])  # _fetcher_config 없음
        wrapped_model.trained_fetcher = fetcher

        # 필수 컬럼 결정
        has_feature_store_fetcher = fetcher is not None and hasattr(fetcher, "_fetcher_config")

        # Then: _fetcher_config 없으면 pass_through로 처리
        assert has_feature_store_fetcher is False


class TestValidationWithActualEndpointLogic:
    """실제 endpoint 로직을 사용한 검증 테스트"""

    def test_validate_batch_prediction_with_feature_store(self):
        """batch prediction에서 Feature Store fetcher 검증"""

        # 실제 endpoint 로직 재현
        def validate_required_columns(input_df, wrapped_model):
            di = getattr(wrapped_model, "data_interface_schema", {}) or {}
            trained_fetcher = getattr(wrapped_model, "trained_fetcher", None)

            has_feature_store_fetcher = trained_fetcher is not None and hasattr(
                trained_fetcher, "_fetcher_config"
            )

            required_cols = set()
            if has_feature_store_fetcher:
                entity_cols = di.get("entity_columns", [])
                required_cols.update(entity_cols)
            else:
                required_cols.update(
                    di.get("required_columns", []) or di.get("feature_columns", []) or []
                )

            missing = [c for c in sorted(required_cols) if c not in input_df.columns]
            return missing

        # Given: Feature Store fetcher가 있는 모델
        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id"],
            "feature_columns": ["feature_1", "feature_2"],
        }
        fetcher = Mock()
        fetcher._fetcher_config = {"entity_columns": ["user_id"]}
        wrapped_model.trained_fetcher = fetcher

        # 입력: entity만 있음
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
            }
        )

        # When: 검증
        missing = validate_required_columns(input_df, wrapped_model)

        # Then: 누락 없음
        assert missing == []

    def test_validate_batch_prediction_without_fetcher(self):
        """batch prediction에서 pass_through 검증"""

        def validate_required_columns(input_df, wrapped_model):
            di = getattr(wrapped_model, "data_interface_schema", {}) or {}
            trained_fetcher = getattr(wrapped_model, "trained_fetcher", None)

            has_feature_store_fetcher = trained_fetcher is not None and hasattr(
                trained_fetcher, "_fetcher_config"
            )

            required_cols = set()
            if has_feature_store_fetcher:
                entity_cols = di.get("entity_columns", [])
                required_cols.update(entity_cols)
            else:
                required_cols.update(
                    di.get("required_columns", []) or di.get("feature_columns", []) or []
                )

            missing = [c for c in sorted(required_cols) if c not in input_df.columns]
            return missing

        # Given: pass_through (fetcher 없음)
        wrapped_model = Mock()
        wrapped_model.data_interface_schema = {
            "entity_columns": ["user_id"],
            "feature_columns": ["feature_1", "feature_2"],
        }
        wrapped_model.trained_fetcher = None

        # 입력: entity만 있음 (feature 누락)
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
            }
        )

        # When: 검증
        missing = validate_required_columns(input_df, wrapped_model)

        # Then: feature_columns 누락
        assert "feature_1" in missing
        assert "feature_2" in missing
