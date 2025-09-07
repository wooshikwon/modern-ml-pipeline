"""
Timeseries 모델들을 sklearn 인터페이스로 감싸는 wrapper들

statsmodels의 시계열 모델들(ARIMA, ExponentialSmoothing)을 
sklearn의 BaseEstimator와 RegressorMixin 인터페이스로 래핑하여
기존 ML 파이프라인에서 사용할 수 있도록 합니다.
"""
import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin

from src.utils.system.logger import logger


class ARIMAWrapper(BaseEstimator, RegressorMixin):
    """
    ARIMA 모델을 sklearn 인터페이스로 감싸는 wrapper
    
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ARIMAWrapper':
        """
        ARIMA 모델 학습
        
        Args:
            X: 특성 데이터 (ARIMA는 univariate이므로 무시됨)
            y: 시계열 타겟 데이터
            
        Returns:
            self: 학습된 ARIMAWrapper 인스턴스
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            logger.info(f"ARIMA 모델 학습 시작 - order=({self.order_p}, {self.order_d}, {self.order_q})")
            
            # ARIMA는 univariate이므로 y만 사용
            self.model_ = ARIMA(y, order=(self.order_p, self.order_d, self.order_q))
            self.fitted_model_ = self.model_.fit()
            self._is_fitted = True
            
            logger.info("✅ ARIMA 모델 학습 완료")
            return self
            
        except ImportError as e:
            logger.error("statsmodels 라이브러리가 설치되어 있지 않습니다. 'pip install statsmodels' 실행 필요")
            raise ImportError("statsmodels is required for ARIMAWrapper") from e
        except Exception as e:
            logger.error(f"ARIMA 모델 학습 실패: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        ARIMA 모델로 예측 수행
        
        Args:
            X: 특성 데이터 (예측할 기간의 길이를 결정하는데 사용)
            
        Returns:
            predictions: 예측 결과 배열
        """
        if not self._is_fitted or self.fitted_model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        try:
            # X의 길이만큼 예측 수행
            forecast_steps = len(X)
            forecast = self.fitted_model_.forecast(steps=forecast_steps)
            
            logger.info(f"ARIMA 예측 완료 - {forecast_steps}개 기간")
            return np.array(forecast)
            
        except Exception as e:
            logger.error(f"ARIMA 예측 실패: {e}")
            raise


class ExponentialSmoothingWrapper(BaseEstimator, RegressorMixin):
    """
    Exponential Smoothing 모델을 sklearn 인터페이스로 감싸는 wrapper
    
    statsmodels.tsa.holtwinters.ExponentialSmoothing을 사용하여 
    지수평활법 기반 시계열 예측을 수행합니다.
    """
    
    def __init__(
        self, 
        trend: Optional[str] = "add", 
        seasonal: Optional[str] = None, 
        seasonal_periods: int = 12
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ExponentialSmoothingWrapper':
        """
        Exponential Smoothing 모델 학습
        
        Args:
            X: 특성 데이터 (무시됨)
            y: 시계열 타겟 데이터
            
        Returns:
            self: 학습된 ExponentialSmoothingWrapper 인스턴스
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            logger.info(f"ExponentialSmoothing 모델 학습 시작 - trend={self.trend}, seasonal={self.seasonal}")
            
            # Seasonal이 설정된 경우에만 seasonal_periods 사용
            periods = self.seasonal_periods if self.seasonal else None
            
            model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=periods
            )
            
            self.fitted_model_ = model.fit()
            self._is_fitted = True
            
            logger.info("✅ ExponentialSmoothing 모델 학습 완료")
            return self
            
        except ImportError as e:
            logger.error("statsmodels 라이브러리가 설치되어 있지 않습니다. 'pip install statsmodels' 실행 필요")
            raise ImportError("statsmodels is required for ExponentialSmoothingWrapper") from e
        except Exception as e:
            logger.error(f"ExponentialSmoothing 모델 학습 실패: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Exponential Smoothing 모델로 예측 수행
        
        Args:
            X: 특성 데이터 (예측할 기간의 길이를 결정하는데 사용)
            
        Returns:
            predictions: 예측 결과 배열
        """
        if not self._is_fitted or self.fitted_model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        try:
            # X의 길이만큼 예측 수행
            forecast_steps = len(X)
            forecast = self.fitted_model_.forecast(steps=forecast_steps)
            
            logger.info(f"ExponentialSmoothing 예측 완료 - {forecast_steps}개 기간")
            return np.array(forecast)
            
        except Exception as e:
            logger.error(f"ExponentialSmoothing 예측 실패: {e}")
            raise


# 클래스들을 모듈에서 바로 접근 가능하도록 export
__all__ = ['ARIMAWrapper', 'ExponentialSmoothingWrapper']