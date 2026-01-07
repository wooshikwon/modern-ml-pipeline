# src/models/custom/tabnet_wrapper.py
"""
TabNet 모델 wrapper - BaseModel 직접 상속

pytorch_tabnet의 TabNetClassifier/TabNetRegressor를 BaseModel 인터페이스로 래핑합니다.
DataFrame 입력을 내부에서 numpy로 변환하여 일관된 인터페이스를 제공합니다.
"""

import warnings
from typing import Any, Optional

# pytorch_tabnet import 전 scipy DeprecationWarning 차단
warnings.filterwarnings("ignore", message=".*scipy.sparse.base.*")
warnings.filterwarnings("ignore", message=".*spmatrix.*")

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.core.logger import logger


class TabNetWrapperBase(BaseModel):
    """TabNet 기반 모델의 공통 베이스 클래스"""

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 0.001,
        momentum: float = 0.02,
        clip_value: float = 1.0,
        optimizer_fn: Any = None,
        optimizer_params: Optional[dict] = None,
        scheduler_fn: Any = None,
        scheduler_params: Optional[dict] = None,
        mask_type: str = "sparsemax",
        seed: int = 42,
        verbose: int = 0,
        **kwargs,
    ):
        """
        TabNet 모델 초기화

        Args:
            n_d: Decision prediction layer 차원
            n_a: Attention embedding 차원
            n_steps: Decision steps 수
            gamma: Feature reusage coefficient
            n_independent: Independent GLU layers 수
            n_shared: Shared GLU layers 수
            lambda_sparse: Sparsity regularization coefficient
            momentum: Batch normalization momentum
            clip_value: Gradient clipping value
            optimizer_fn: Optimizer 클래스
            optimizer_params: Optimizer 파라미터
            scheduler_fn: Scheduler 클래스
            scheduler_params: Scheduler 파라미터
            mask_type: Mask type ('sparsemax' or 'entmax')
            seed: Random seed
            verbose: Verbosity level
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params or {"lr": 2e-2}
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.seed = seed
        self.verbose = verbose
        self.extra_kwargs = kwargs

        self.model = None
        self.is_fitted = False

        logger.info("[INIT:TabNet] 초기화 완료")
        logger.info(f"  구조: n_d={n_d}, n_a={n_a}, n_steps={n_steps}")

    def _get_model_params(self) -> dict:
        """TabNet 모델 생성 파라미터 반환"""
        params = {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "n_independent": self.n_independent,
            "n_shared": self.n_shared,
            "lambda_sparse": self.lambda_sparse,
            "momentum": self.momentum,
            "clip_value": self.clip_value,
            "optimizer_params": self.optimizer_params,
            "mask_type": self.mask_type,
            "seed": self.seed,
            "verbose": self.verbose,
        }
        if self.optimizer_fn:
            params["optimizer_fn"] = self.optimizer_fn
        if self.scheduler_fn:
            params["scheduler_fn"] = self.scheduler_fn
            params["scheduler_params"] = self.scheduler_params
        return params

    def _to_numpy(self, X, y=None):
        """DataFrame/Series를 numpy array로 변환"""
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = None
        if y is not None:
            y_np = y.values if isinstance(y, pd.Series) else y
        return X_np, y_np

    def set_params(self, **params):
        """하이퍼파라미터 설정"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        """하이퍼파라미터 반환"""
        return {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "n_independent": self.n_independent,
            "n_shared": self.n_shared,
            "lambda_sparse": self.lambda_sparse,
            "momentum": self.momentum,
            "clip_value": self.clip_value,
            "mask_type": self.mask_type,
            "seed": self.seed,
            "verbose": self.verbose,
        }


class TabNetClassifierWrapper(TabNetWrapperBase):
    """TabNet 분류 모델 wrapper"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_classes = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "TabNetClassifierWrapper":
        """
        분류 모델 학습

        Args:
            X: 입력 DataFrame
            y: 타겟 Series

        Returns:
            학습된 모델 인스턴스
        """
        from pytorch_tabnet.tab_model import TabNetClassifier

        logger.info("[TRAIN:TabNet] Classification 학습 시작")

        # DataFrame → numpy 변환
        X_np, y_np = self._to_numpy(X, y)

        # 클래스 수 저장
        self._n_classes = len(np.unique(y_np))

        # 모델 생성
        self.model = TabNetClassifier(**self._get_model_params())

        # 학습
        self.model.fit(X_np, y_np)
        self.is_fitted = True

        logger.info(f"[TRAIN:TabNet] 학습 완료 - {X_np.shape[0]}샘플, {self._n_classes}클래스")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        분류 예측 수행

        Args:
            X: 입력 DataFrame

        Returns:
            예측 클래스 라벨 배열
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_np, _ = self._to_numpy(X)
        return self.model.predict(X_np)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        분류 확률 예측 수행

        Args:
            X: 입력 DataFrame

        Returns:
            예측 확률 배열 - shape: (n_samples, n_classes)
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_np, _ = self._to_numpy(X)
        return self.model.predict_proba(X_np)


class TabNetRegressorWrapper(TabNetWrapperBase):
    """TabNet 회귀 모델 wrapper"""

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "TabNetRegressorWrapper":
        """
        회귀 모델 학습

        Args:
            X: 입력 DataFrame
            y: 타겟 Series

        Returns:
            학습된 모델 인스턴스
        """
        from pytorch_tabnet.tab_model import TabNetRegressor

        logger.info("[TRAIN:TabNet] Regression 학습 시작")

        # DataFrame → numpy 변환
        X_np, y_np = self._to_numpy(X, y)

        # y를 2D로 변환 (TabNetRegressor 요구사항)
        if y_np is not None and y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        # 모델 생성
        self.model = TabNetRegressor(**self._get_model_params())

        # 학습
        self.model.fit(X_np, y_np)
        self.is_fitted = True

        logger.info(f"[TRAIN:TabNet] 학습 완료 - {X_np.shape[0]}샘플")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        회귀 예측 수행

        Args:
            X: 입력 DataFrame

        Returns:
            예측값 배열
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_np, _ = self._to_numpy(X)
        predictions = self.model.predict(X_np)

        # 1D로 변환
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        return predictions


__all__ = ["TabNetClassifierWrapper", "TabNetRegressorWrapper", "TabNetWrapperBase"]
