"""
Temporal split 기능 테스트.

TabularDataHandler의 temporal 분할이 시간순으로 올바르게 동작하는지,
Recipe 스키마 검증이 잘못된 설정을 올바르게 거부하는지 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from mmp.components.datahandler.modules.tabular_handler import TabularDataHandler
from mmp.settings.recipe import DataSplit


class TestTemporalSplit:
    """TabularDataHandler temporal split 동작 검증"""

    def test_temporal_split_preserves_time_order(self, settings_builder):
        """temporal 분할이 시간순을 유지하는지 검증"""
        settings = (
            settings_builder.with_task("regression").with_target_column("target").build()
        )
        settings.recipe.data.split.strategy = "temporal"
        settings.recipe.data.split.temporal_column = "created_at"
        settings.recipe.data.data_interface.feature_columns = ["f1", "f2"]

        # 시간순이 아닌 데이터 생성
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "created_at": pd.date_range("2025-01-01", periods=n, freq="D"),
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "target": np.random.randn(n),
        })
        df = df.sample(frac=1, random_state=0)  # 셔플

        handler = TabularDataHandler(settings)
        result = handler.split_data(df)

        # train의 최대 날짜 < validation의 최소 날짜
        assert result["train"]["created_at"].max() < result["validation"]["created_at"].min()
        # validation의 최대 날짜 < test의 최소 날짜
        assert result["validation"]["created_at"].max() < result["test"]["created_at"].min()

    def test_temporal_split_ratios(self, settings_builder):
        """temporal 분할 비율이 설정과 일치하는지 검증"""
        settings = (
            settings_builder.with_task("regression").with_target_column("target").build()
        )
        settings.recipe.data.split.strategy = "temporal"
        settings.recipe.data.split.temporal_column = "ts"
        settings.recipe.data.split.train = 0.6
        settings.recipe.data.split.validation = 0.2
        settings.recipe.data.split.test = 0.2
        settings.recipe.data.data_interface.feature_columns = ["f1"]

        n = 1000
        df = pd.DataFrame({
            "ts": range(n),
            "f1": np.random.randn(n),
            "target": np.random.randn(n),
        })

        handler = TabularDataHandler(settings)
        result = handler.split_data(df)

        assert len(result["train"]) == 600
        assert len(result["validation"]) == 200
        assert len(result["test"]) == 200

    def test_temporal_split_with_calibration(self, settings_builder):
        """calibration 포함 temporal 분할 검증"""
        settings = (
            settings_builder.with_task("regression").with_target_column("target").build()
        )
        settings.recipe.data.split.strategy = "temporal"
        settings.recipe.data.split.temporal_column = "ts"
        settings.recipe.data.split.train = 0.5
        settings.recipe.data.split.validation = 0.2
        settings.recipe.data.split.test = 0.2
        settings.recipe.data.split.calibration = 0.1
        settings.recipe.data.data_interface.feature_columns = ["f1"]

        n = 1000
        df = pd.DataFrame({
            "ts": range(n),
            "f1": np.random.randn(n),
            "target": np.random.randn(n),
        })

        handler = TabularDataHandler(settings)
        result = handler.split_data(df)

        assert len(result["train"]) == 500
        assert len(result["validation"]) == 200
        assert result["calibration"] is not None
        assert len(result["calibration"]) == 100
        # calibration이 가장 마지막 시간대
        assert result["test"]["ts"].max() < result["calibration"]["ts"].min()

    def test_temporal_split_missing_column_raises(self, settings_builder):
        """존재하지 않는 temporal_column이면 에러"""
        settings = (
            settings_builder.with_task("regression").with_target_column("target").build()
        )
        settings.recipe.data.split.strategy = "temporal"
        settings.recipe.data.split.temporal_column = "nonexistent"
        settings.recipe.data.data_interface.feature_columns = ["f1"]

        df = pd.DataFrame({"f1": [1, 2, 3], "target": [1, 2, 3]})
        handler = TabularDataHandler(settings)

        with pytest.raises(ValueError, match="temporal_column 'nonexistent'"):
            handler.split_data(df)

    def test_random_split_unchanged(self, settings_builder):
        """기존 random split이 영향받지 않는지 검증"""
        settings = (
            settings_builder.with_task("regression").with_target_column("target").build()
        )
        # strategy 미설정 = 기본값 random
        settings.recipe.data.data_interface.feature_columns = ["f1"]

        n = 100
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "target": np.random.randn(n),
        })

        handler = TabularDataHandler(settings)
        result = handler.split_data(df)

        total = len(result["train"]) + len(result["validation"]) + len(result["test"])
        assert total == n


class TestDataSplitSchema:
    """DataSplit Pydantic 스키마 검증"""

    def test_default_strategy_is_random(self):
        split = DataSplit(train=0.6, validation=0.2, test=0.2)
        assert split.strategy == "random"

    def test_temporal_requires_column(self):
        with pytest.raises(ValueError, match="temporal_column이 필수"):
            DataSplit(strategy="temporal", train=0.6, validation=0.2, test=0.2)

    def test_temporal_with_column_valid(self):
        split = DataSplit(
            strategy="temporal", temporal_column="created_at",
            train=0.6, validation=0.2, test=0.2,
        )
        assert split.temporal_column == "created_at"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError, match="지원하지 않는 분할 전략"):
            DataSplit(strategy="invalid", train=0.6, validation=0.2, test=0.2)
