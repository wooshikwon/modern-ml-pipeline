import pandas as pd
import pytest

from src.factory import Factory


class TestPyfuncWrapperSignature:
    """PyfuncWrapper 시그니처/스키마 생성 검증.

    목적: fetcher.timestamp_column이 스키마 메타데이터에 반영되고
    datetime 캐스팅이 적용되는지 확인한다.
    """

    @pytest.mark.skip(reason="Feast RepoConfig Pydantic v2 호환성 이슈")
    def test_signature_uses_fetcher_timestamp_column_and_casts_datetime(self, settings_builder):
        # Skip if feast not installed
        try:
            import feast  # noqa: F401
        except ImportError:
            pytest.skip("feast not installed; skipping feature_store fetcher test")

        # Settings: timeseries + feature_store + timestamp_column
        settings = (
            settings_builder.with_task("timeseries")
            .with_model("sklearn.linear_model.LinearRegression")
            .with_target_column("target")
            .with_entity_columns(["entity_id"])
            .with_timestamp_column("timestamp")
            .with_feature_store(enabled=True)
            .build()
        )

        factory = Factory(settings)
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        model = factory.create_model()

        # Training DataFrame: timestamp는 문자열로 제공하여 캐스팅 여부 검증
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "entity_id": [1, 1, 1, 1, 1],
                "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [1.0, 0.9, 1.1, 1.05, 0.95],
            }
        )

        wrapper = factory.create_pyfunc_wrapper(
            trained_model=model,
            trained_datahandler=datahandler,
            trained_preprocessor=None,
            trained_fetcher=fetcher,
            training_df=df,
            training_results={},
        )

        # Assertions
        assert wrapper.signature is not None
        assert wrapper.data_schema is not None

        schema = wrapper.data_schema
        # timestamp_column 메타데이터 반영
        assert schema.get("timestamp_column") == "timestamp"

        # column_types에 timestamp가 존재하며 datetime으로 캐스팅되었는지 확인
        column_types = schema.get("column_types", {})
        assert "timestamp" in column_types
        assert "datetime" in column_types["timestamp"].lower()
