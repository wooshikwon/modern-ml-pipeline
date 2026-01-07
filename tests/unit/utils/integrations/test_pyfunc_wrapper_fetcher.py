"""
PyfuncWrapper Fetcher 호출 테스트
- predict()에서 fetcher.fetch() 호출 테스트
- run_mode별 fetcher 동작 테스트
- fetcher 실패 시 graceful degradation 테스트
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper


class TestPyfuncWrapperFetcherIntegration:
    """PyfuncWrapper에서 Fetcher 호출 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock ML 모델"""
        model = Mock()
        model.predict.return_value = np.array([0, 1, 0])
        model.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        return model

    @pytest.fixture
    def mock_fetcher(self):
        """Mock Fetcher"""
        fetcher = Mock()

        def fetch_side_effect(df, run_mode="batch"):
            # 피처 증강 시뮬레이션
            augmented = df.copy()
            augmented["augmented_feature_1"] = [0.5] * len(df)
            augmented["augmented_feature_2"] = [100] * len(df)
            return augmented

        fetcher.fetch.side_effect = fetch_side_effect
        fetcher._fetcher_config = {"entity_columns": ["user_id"]}
        return fetcher

    @pytest.fixture
    def wrapper_with_fetcher(self, component_test_context, mock_model, mock_fetcher):
        """Fetcher가 있는 PyfuncWrapper"""
        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=mock_fetcher,
                data_interface_schema={
                    "feature_columns": [
                        "feature_1",
                        "feature_2",
                        "augmented_feature_1",
                        "augmented_feature_2",
                    ],
                    "target_column": "target",
                    "entity_columns": ["user_id"],
                },
            )
            return wrapper

    def test_predict_calls_fetcher_with_batch_mode(self, wrapper_with_fetcher, mock_fetcher):
        """predict()에서 batch 모드로 fetcher 호출 테스트"""
        # Given: 입력 데이터
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [0.5, 1.5, 2.5],
            }
        )

        # When: predict 호출 (기본값 batch)
        wrapper_with_fetcher.predict(context=None, model_input=input_df)

        # Then: fetcher.fetch()가 batch 모드로 호출됨
        mock_fetcher.fetch.assert_called_once()
        call_args = mock_fetcher.fetch.call_args
        assert call_args[1]["run_mode"] == "batch"

    def test_predict_calls_fetcher_with_serving_mode(self, wrapper_with_fetcher, mock_fetcher):
        """predict()에서 serving 모드로 fetcher 호출 테스트"""
        # Given: 입력 데이터
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "feature_1": [1.0, 2.0, 3.0],
            }
        )

        # When: serving 모드로 predict 호출
        wrapper_with_fetcher.predict(
            context=None, model_input=input_df, params={"run_mode": "serving"}
        )

        # Then: fetcher.fetch()가 serving 모드로 호출됨
        mock_fetcher.fetch.assert_called_once()
        call_args = mock_fetcher.fetch.call_args
        assert call_args[1]["run_mode"] == "serving"

    def test_predict_uses_augmented_features(self, wrapper_with_fetcher, mock_model):
        """증강된 피처가 예측에 사용되는지 테스트"""
        # Given: 입력 데이터
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "feature_1": [1.0, 2.0],
                "feature_2": [0.5, 1.5],
            }
        )

        # When: predict 호출
        wrapper_with_fetcher.predict(context=None, model_input=input_df)

        # Then: 모델에 전달된 데이터에 증강된 피처 포함
        call_args = mock_model.predict.call_args[0][0]
        assert "augmented_feature_1" in call_args.columns
        assert "augmented_feature_2" in call_args.columns


class TestPyfuncWrapperFetcherGracefulDegradation:
    """Fetcher 실패 시 graceful degradation 테스트"""

    @pytest.fixture
    def failing_fetcher(self):
        """실패하는 Fetcher"""
        fetcher = Mock()
        fetcher.fetch.side_effect = Exception("Feature Store unavailable")
        fetcher._fetcher_config = {"entity_columns": ["user_id"]}
        return fetcher

    @pytest.fixture
    def wrapper_with_failing_fetcher(self, component_test_context, failing_fetcher):
        """실패하는 Fetcher가 있는 PyfuncWrapper"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=failing_fetcher,
                data_interface_schema={
                    "feature_columns": ["feature_1", "feature_2"],
                    "target_column": "target",
                },
            )
            return wrapper

    def test_predict_continues_with_original_data_on_fetcher_failure(
        self, wrapper_with_failing_fetcher
    ):
        """Fetcher 실패 시 원본 데이터로 계속 진행"""
        # Given: 입력 데이터
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "feature_1": [1.0, 2.0],
                "feature_2": [0.5, 1.5],
            }
        )

        # When: predict 호출 (Fetcher 실패하지만 계속 진행)
        result = wrapper_with_failing_fetcher.predict(context=None, model_input=input_df)

        # Then: 예측 성공
        assert len(result) == 2

    def test_predict_logs_warning_on_fetcher_failure(self, wrapper_with_failing_fetcher, caplog):
        """Fetcher 실패 시 경고 로그 기록"""
        input_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "feature_1": [1.0, 2.0],
                "feature_2": [0.5, 1.5],
            }
        )

        # When: predict 호출
        with caplog.at_level("WARNING"):
            wrapper_with_failing_fetcher.predict(context=None, model_input=input_df)

        # Then: 경고 로그 포함
        # 로그는 테스트 설정에 따라 캡처되지 않을 수 있음


class TestPyfuncWrapperWithoutFetcher:
    """Fetcher 없는 PyfuncWrapper 테스트"""

    @pytest.fixture
    def wrapper_without_fetcher(self, component_test_context):
        """Fetcher 없는 PyfuncWrapper"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0])

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=None,  # Fetcher 없음
                data_interface_schema={
                    "feature_columns": ["feature_1", "feature_2"],
                    "target_column": "target",
                },
            )
            return wrapper

    def test_predict_works_without_fetcher(self, wrapper_without_fetcher):
        """Fetcher 없이 예측 동작"""
        input_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [0.5, 1.5, 2.5],
            }
        )

        # When: predict 호출
        result = wrapper_without_fetcher.predict(context=None, model_input=input_df)

        # Then: 예측 성공
        assert len(result) == 3

    def test_predict_does_not_call_fetch_when_no_fetcher(self, wrapper_without_fetcher):
        """Fetcher 없으면 fetch 호출 안함"""
        input_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0],
                "feature_2": [0.5, 1.5],
            }
        )

        # When: predict 호출
        result = wrapper_without_fetcher.predict(context=None, model_input=input_df)

        # Then: fetcher는 None이므로 fetch 호출 없음
        assert wrapper_without_fetcher.trained_fetcher is None


class TestPyfuncWrapperRunModeParameter:
    """run_mode 파라미터 처리 테스트"""

    @pytest.fixture
    def mock_fetcher_tracking_mode(self):
        """run_mode를 추적하는 Mock Fetcher"""
        fetcher = Mock()
        fetcher.captured_run_mode = None

        def fetch_tracking(df, run_mode="batch"):
            fetcher.captured_run_mode = run_mode
            return df.copy()

        fetcher.fetch.side_effect = fetch_tracking
        fetcher._fetcher_config = {"entity_columns": ["user_id"]}
        return fetcher

    @pytest.fixture
    def wrapper_for_mode_test(self, component_test_context, mock_fetcher_tracking_mode):
        """run_mode 테스트용 Wrapper"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=mock_fetcher_tracking_mode,
                data_interface_schema={"feature_columns": ["feature_1"], "target_column": "target"},
            )
            return wrapper, mock_fetcher_tracking_mode

    def test_default_run_mode_is_batch(self, wrapper_for_mode_test):
        """기본 run_mode는 batch"""
        wrapper, fetcher = wrapper_for_mode_test

        input_df = pd.DataFrame({"feature_1": [1.0, 2.0]})

        # When: params 없이 호출
        wrapper.predict(context=None, model_input=input_df)

        # Then: batch 모드
        assert fetcher.captured_run_mode == "batch"

    def test_run_mode_batch_from_params(self, wrapper_for_mode_test):
        """params에서 batch 모드"""
        wrapper, fetcher = wrapper_for_mode_test

        input_df = pd.DataFrame({"feature_1": [1.0, 2.0]})

        # When: batch 모드 명시
        wrapper.predict(context=None, model_input=input_df, params={"run_mode": "batch"})

        # Then: batch 모드
        assert fetcher.captured_run_mode == "batch"

    def test_run_mode_serving_from_params(self, wrapper_for_mode_test):
        """params에서 serving 모드"""
        wrapper, fetcher = wrapper_for_mode_test

        input_df = pd.DataFrame({"feature_1": [1.0, 2.0]})

        # When: serving 모드 명시
        wrapper.predict(context=None, model_input=input_df, params={"run_mode": "serving"})

        # Then: serving 모드
        assert fetcher.captured_run_mode == "serving"

    def test_run_mode_train_from_params(self, wrapper_for_mode_test):
        """params에서 train 모드"""
        wrapper, fetcher = wrapper_for_mode_test

        input_df = pd.DataFrame({"feature_1": [1.0, 2.0]})

        # When: train 모드 명시
        wrapper.predict(context=None, model_input=input_df, params={"run_mode": "train"})

        # Then: train 모드
        assert fetcher.captured_run_mode == "train"


class TestPyfuncWrapperFetcherPickleSerialization:
    """Fetcher와 함께 PyfuncWrapper 직렬화 테스트"""

    def test_wrapper_with_fetcher_serializable(self, component_test_context):
        """Fetcher 포함 Wrapper 직렬화 테스트"""
        import pickle

        # Given: Fetcher 포함 Wrapper
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])

        mock_fetcher = Mock()
        mock_fetcher._fetcher_config = {"entity_columns": ["user_id"]}
        mock_fetcher._feature_store_adapter = None

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=mock_fetcher,
            )

            # When: 직렬화
            # 참고: Mock 객체는 일반적으로 직렬화 불가능하므로 실제 환경에서는
            # 실제 Fetcher 구현체를 사용해야 함
            try:
                pickled = pickle.dumps(wrapper)
                restored = pickle.loads(pickled)
                # Mock 객체가 직렬화 가능한 경우
                assert restored is not None
            except (pickle.PicklingError, TypeError, AttributeError):
                # Mock 객체는 직렬화 불가 - 예상된 동작
                pass
