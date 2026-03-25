import importlib
from typing import Any

import numpy as np
import pandas as pd

from mmp.models.base import BaseModel
from mmp.utils.core.logger import logger

_LIBRARY_OBJECTIVE_MAP = {
    "lightgbm": lambda q: {"objective": "quantile", "alpha": q},
    "xgboost": lambda q: {"objective": "reg:quantileerror", "quantile_alpha": q},
    "catboost": lambda q: {"loss_function": f"Quantile:alpha={q}"},
}


class QuantileRegressorEnsemble(BaseModel):
    """Quantile regression ensemble using multiple gradient boosting models."""

    def __init__(self, base_class_path: str, quantiles: list[float], **kwargs: Any):
        self.base_class_path = base_class_path
        self.quantiles = quantiles
        self.base_params = dict(kwargs)
        self.models: dict[float, Any] = {}
        self.is_fitted = False

        logger.info("[INIT:QuantileEnsemble] 초기화 완료")
        logger.info(f"  base={base_class_path}, quantiles={quantiles}")

    def _resolve_class_path(self) -> str:
        """base_class_path가 문자열이면 그대로, 클래스 객체면 모듈 경로로 변환."""
        if isinstance(self.base_class_path, str):
            return self.base_class_path
        # model_factory._process_hyperparameters가 _class 키를 실제 클래스로 변환하는 경우
        cls = self.base_class_path
        return f"{cls.__module__}.{cls.__name__}"

    def _detect_library(self) -> str:
        class_path = self._resolve_class_path()
        module_path = class_path.rsplit(".", 1)[0]
        for lib_key in _LIBRARY_OBJECTIVE_MAP:
            if lib_key in module_path:
                return lib_key
        raise ValueError(f"Unsupported library for quantile regression: {module_path}")

    def _load_class(self):
        if not isinstance(self.base_class_path, str):
            return self.base_class_path
        module_path, class_name = self.base_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _to_numpy(X, y=None):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = None
        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else y
        return X_np, y_np

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "QuantileRegressorEnsemble":
        model_class = self._load_class()
        library = self._detect_library()
        obj_fn = _LIBRARY_OBJECTIVE_MAP[library]

        X_np, y_np = self._to_numpy(X, y)

        for q in self.quantiles:
            q_params = {**self.base_params, **obj_fn(q)}
            model = model_class(**q_params)
            model.fit(X_np, y_np)
            self.models[q] = model
            logger.info(f"[TRAIN:QuantileEnsemble] quantile={q} 학습 완료")

        self.is_fitted = True
        logger.info(f"[TRAIN:QuantileEnsemble] 전체 학습 완료 - {len(self.quantiles)}개 분위수")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_np, _ = self._to_numpy(X)
        results = {}

        for q, model in self.models.items():
            preds = model.predict(X_np)
            if isinstance(preds, pd.DataFrame):
                preds = preds.values.flatten()
            elif isinstance(preds, np.ndarray) and preds.ndim > 1:
                preds = preds.flatten()
            col_name = f"pred_p{int(q * 100)}"
            results[col_name] = preds

        return pd.DataFrame(results)

    def get_params(self, deep: bool = True) -> dict:
        params = {
            "base_class_path": self.base_class_path,
            "quantiles": self.quantiles,
        }
        params.update(self.base_params)
        return params

    def set_params(self, **params: Any) -> "QuantileRegressorEnsemble":
        for key, value in params.items():
            if key in ("base_class_path", "quantiles"):
                setattr(self, key, value)
            else:
                self.base_params[key] = value
        return self
