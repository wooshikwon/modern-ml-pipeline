"""
Timeseries 모델들을 BaseModel 인터페이스로 감싸는 wrapper들

statsmodels의 시계열 모델들(ARIMA, ExponentialSmoothing)을
BaseModel 인터페이스로 래핑하여 ML 파이프라인에서 사용할 수 있도록 합니다.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.core.logger import logger


class ARIMA(BaseModel):
    """
    ARIMA 모델을 BaseModel 인터페이스로 감싸는 wrapper

    statsmodels.tsa.arima.model.ARIMA를 사용하여 시계열 예측을 수행합니다.
    ARIMA는 univariate 모델이므로 fit() 시 y만 사용하고, X는 무시됩니다.
    """

    def __init__(self, order_p: int = 1, order_d: int = 1, order_q: int = 1):
        """
        ARIMA 모델 초기화

        Args:
            order_p: AR(p) - 자기회귀 차수
            order_d: I(d) - 차분 차수
            order_q: MA(q) - 이동평균 차수
        """
        self.order_p = order_p
        self.order_d = order_d
        self.order_q = order_q
        self.model_ = None
        self.fitted_model_ = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "ARIMA":
        """
        ARIMA 모델 학습

        Args:
            X: 특성 데이터 (ARIMA는 univariate이므로 무시됨)
            y: 시계열 타겟 데이터
            **kwargs: 추가 인자

        Returns:
            self: 학습된 ARIMA 인스턴스
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

            logger.info(
                f"[TRAIN:ARIMA] 학습 시작 - order=({self.order_p}, {self.order_d}, {self.order_q})"
            )

            # ARIMA는 univariate이므로 y만 사용
            self.model_ = StatsARIMA(y, order=(self.order_p, self.order_d, self.order_q))
            self.fitted_model_ = self.model_.fit()
            self._is_fitted = True

            logger.info("[TRAIN:ARIMA] 학습 완료")
            return self

        except ImportError as e:
            logger.error("[TRAIN:ARIMA] statsmodels 라이브러리 미설치")
            raise ImportError("statsmodels is required for ARIMA") from e
        except Exception as e:
            logger.error(f"[TRAIN:ARIMA] 학습 실패: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ARIMA 모델로 예측 수행

        Args:
            X: 특성 데이터 (예측할 기간의 길이를 결정하는데 사용)

        Returns:
            pd.DataFrame: 예측 결과 ('prediction' 컬럼 포함)
        """
        if not self._is_fitted or self.fitted_model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")

        try:
            forecast_steps = len(X)
            forecast = self.fitted_model_.forecast(steps=forecast_steps)

            logger.info(f"[INFER:ARIMA] 예측 완료 - {forecast_steps}개 기간")
            return pd.DataFrame({"prediction": np.array(forecast)}, index=X.index)

        except Exception as e:
            logger.error(f"[INFER:ARIMA] 예측 실패: {e}")
            raise


class ExponentialSmoothing(BaseModel):
    """
    Exponential Smoothing 모델을 BaseModel 인터페이스로 감싸는 wrapper

    statsmodels.tsa.holtwinters.ExponentialSmoothing을 사용하여
    지수평활법 기반 시계열 예측을 수행합니다.
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: int = 12,
    ):
        """
        Exponential Smoothing 모델 초기화

        Args:
            trend: 트렌드 성분 ("add", "mul", None)
            seasonal: 계절성 성분 ("add", "mul", None)
            seasonal_periods: 계절성 주기 (월별=12, 주별=52 등)
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model_ = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "ExponentialSmoothing":
        """
        Exponential Smoothing 모델 학습

        Args:
            X: 특성 데이터 (무시됨)
            y: 시계열 타겟 데이터
            **kwargs: 추가 인자

        Returns:
            self: 학습된 ExponentialSmoothing 인스턴스
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as StatsES

            logger.info(
                f"[TRAIN:ExpSmoothing] 학습 시작 - trend={self.trend}, seasonal={self.seasonal}"
            )

            periods = self.seasonal_periods if self.seasonal else None

            model = StatsES(y, trend=self.trend, seasonal=self.seasonal, seasonal_periods=periods)

            self.fitted_model_ = model.fit()
            self._is_fitted = True

            logger.info("[TRAIN:ExpSmoothing] 학습 완료")
            return self

        except ImportError as e:
            logger.error("[TRAIN:ExpSmoothing] statsmodels 라이브러리 미설치")
            raise ImportError("statsmodels is required for ExponentialSmoothing") from e
        except Exception as e:
            logger.error(f"[TRAIN:ExpSmoothing] 학습 실패: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Exponential Smoothing 모델로 예측 수행

        Args:
            X: 특성 데이터 (예측할 기간의 길이를 결정하는데 사용)

        Returns:
            pd.DataFrame: 예측 결과 ('prediction' 컬럼 포함)
        """
        if not self._is_fitted or self.fitted_model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")

        try:
            forecast_steps = len(X)
            forecast = self.fitted_model_.forecast(steps=forecast_steps)

            logger.info(f"[INFER:ExpSmoothing] 예측 완료 - {forecast_steps}개 기간")
            return pd.DataFrame({"prediction": np.array(forecast)}, index=X.index)

        except Exception as e:
            logger.error(f"[INFER:ExpSmoothing] 예측 실패: {e}")
            raise


__all__ = ["ARIMA", "ExponentialSmoothing"]
