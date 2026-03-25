"""
QuantileRegressorEnsemble Tests

tests/README.md 테스트 전략 준수:
- 퍼블릭 API만 호출
- 결정론적 테스트 (고정 시드)
- 실제 데이터 흐름 검증
"""

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="LightGBM not installed")

from mmp.models.custom.quantile_ensemble import QuantileRegressorEnsemble


class TestQuantileRegressorEnsembleInit:
    """초기화 및 파라미터 관리 테스트"""

    def test_init_stores_params(self):
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5, 0.75, 0.95],
            num_leaves=31,
            learning_rate=0.1,
        )
        assert model.base_class_path == "lightgbm.LGBMRegressor"
        assert model.quantiles == [0.5, 0.75, 0.95]
        assert model.base_params == {"num_leaves": 31, "learning_rate": 0.1}
        assert model.is_fitted is False

    def test_get_params_flattens_base_params(self):
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5, 0.9],
            num_leaves=50,
        )
        params = model.get_params()
        assert params["base_class_path"] == "lightgbm.LGBMRegressor"
        assert params["quantiles"] == [0.5, 0.9]
        assert params["num_leaves"] == 50

    def test_set_params_routes_correctly(self):
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
        )
        result = model.set_params(num_leaves=100, learning_rate=0.05)
        assert result is model  # returns self
        assert model.base_params["num_leaves"] == 100
        assert model.base_params["learning_rate"] == 0.05

    def test_set_params_updates_quantiles(self):
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
        )
        model.set_params(quantiles=[0.25, 0.5, 0.75])
        assert model.quantiles == [0.25, 0.5, 0.75]

    def test_predict_before_fit_raises(self):
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
        )
        with pytest.raises(RuntimeError, match="학습되지 않았습니다"):
            model.predict(pd.DataFrame({"a": [1, 2, 3]}))

    def test_unsupported_library_raises(self):
        model = QuantileRegressorEnsemble(
            base_class_path="sklearn.linear_model.LinearRegression",
            quantiles=[0.5],
        )
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="Unsupported library"):
            model.fit(X, y)


class TestQuantileRegressorEnsembleFitPredict:
    """학습 및 예측 테스트 - LightGBM 기반"""

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        })
        y = pd.Series(3 * X["feat_1"] + 2 * X["feat_2"] + np.random.randn(n) * 0.5)
        return X, y

    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
            verbose=-1,
        )
        result = model.fit(X, y)
        assert result is model
        assert model.is_fitted is True

    def test_predict_returns_correct_columns(self, regression_data):
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5, 0.75, 0.95, 0.99],
            verbose=-1,
            n_estimators=20,
        )
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])

        assert isinstance(preds, pd.DataFrame)
        assert list(preds.columns) == ["pred_p50", "pred_p75", "pred_p95", "pred_p99"]
        assert len(preds) == 50

    def test_quantile_ordering(self, regression_data):
        """Higher quantiles should produce higher predictions on average."""
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.1, 0.5, 0.9],
            verbose=-1,
            n_estimators=50,
        )
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])

        assert preds["pred_p10"].mean() < preds["pred_p50"].mean()
        assert preds["pred_p50"].mean() < preds["pred_p90"].mean()

    def test_single_quantile(self, regression_data):
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
            verbose=-1,
            n_estimators=10,
        )
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])

        assert list(preds.columns) == ["pred_p50"]
        assert not preds["pred_p50"].isna().any()

    def test_fit_creates_model_per_quantile(self, regression_data):
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.25, 0.5, 0.75],
            verbose=-1,
            n_estimators=10,
        )
        model.fit(X, y)
        assert len(model.models) == 3
        assert set(model.models.keys()) == {0.25, 0.5, 0.75}

    def test_set_params_affects_subsequent_fit(self, regression_data):
        X, y = regression_data
        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5],
            verbose=-1,
            n_estimators=5,
        )
        model.set_params(n_estimators=50)
        model.fit(X[:150], y[:150])

        # Verify the internal model used updated params
        inner_model = model.models[0.5]
        assert inner_model.n_estimators == 50


class TestQuantileRegressorEnsemblePickle:
    """직렬화 테스트 - MLflow artifact 저장 호환성"""

    def test_pickle_roundtrip(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = pd.Series(X["a"] * 2 + np.random.randn(100) * 0.3)

        model = QuantileRegressorEnsemble(
            base_class_path="lightgbm.LGBMRegressor",
            quantiles=[0.5, 0.9],
            verbose=-1,
            n_estimators=10,
        )
        model.fit(X[:80], y[:80])
        preds_before = model.predict(X[80:])

        # Pickle roundtrip
        data = pickle.dumps(model)
        loaded = pickle.loads(data)

        preds_after = loaded.predict(X[80:])

        pd.testing.assert_frame_equal(preds_before, preds_after)
        assert loaded.is_fitted is True
        assert loaded.base_class_path == "lightgbm.LGBMRegressor"
        assert loaded.quantiles == [0.5, 0.9]
