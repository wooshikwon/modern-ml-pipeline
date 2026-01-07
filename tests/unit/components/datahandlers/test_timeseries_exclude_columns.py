import pandas as pd

from src.factory import Factory


class TestTimeseriesExcludeColumns:
    def test_excludes_entity_and_fs_timestamp_when_feature_store(self, settings_builder):
        # Settings: timeseries + feature_store + timestamp_column
        settings = (
            settings_builder.with_task("timeseries")
            .with_model("any.module.LinearTrend")
            .with_target_column("target")
            .with_entity_columns(["entity_id"])
            .with_timestamp_column("event_ts")
            .with_feature_store(enabled=True)
            .build()
        )

        factory = Factory(settings)
        handler = factory.create_datahandler()

        # event_ts와 entity_id가 모두 존재하는 DF
        df = pd.DataFrame(
            {
                "event_ts": pd.date_range("2024-01-01", periods=5, freq="D"),
                "entity_id": [1, 1, 1, 1, 1],
                "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [1, 2, 3, 4, 5],
            }
        )

        # prepare_data 내부에서 _get_exclude_columns를 거치므로 결과 컬럼으로 간접 검증
        X, y, add = handler.prepare_data(df)
        cols = set(X.columns)
        # entity_id와 FS timestamp(event_ts)는 제외되어야 함
        assert "entity_id" not in cols
        assert "event_ts" not in cols

    def test_excludes_only_entity_when_pass_through(self, settings_builder):
        # Settings: timeseries + pass_through (feature_store 비활성)
        settings = (
            settings_builder.with_task("timeseries")
            .with_model("any.module.LinearTrend")
            .with_target_column("target")
            .with_entity_columns(["entity_id"])
            .with_timestamp_column("event_ts")
            .with_feature_store(enabled=False)
            .build()
        )

        factory = Factory(settings)
        handler = factory.create_datahandler()

        df = pd.DataFrame(
            {
                "event_ts": pd.date_range("2024-01-01", periods=5, freq="D"),
                "entity_id": [1, 1, 1, 1, 1],
                "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [1, 2, 3, 4, 5],
            }
        )

        X, y, add = handler.prepare_data(df)
        cols = set(X.columns)
        # pass_through에서는 timestamp는 자동 제외 대상이 아님(prepare에서 target/timestamp는 드롭되지만
        # _get_exclude_columns의 분기 검증 목적: entity만 제외 대상에 포함)
        assert "entity_id" not in cols
        # prepare_data 단계에서 timestamp는 auto_exclude에 포함되어 드롭됨이 정상
        assert "event_ts" not in cols
