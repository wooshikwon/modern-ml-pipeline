"""
BaseCalibrator - 확률 보정기 기본 인터페이스
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseCalibrator(ABC):
    """
    모든 Calibrator가 따라야 할 표준 계약.
    분류 모델의 예측 확률을 보정하는 Strategy Pattern의 기반.
    """

    def __init__(self):
        """Calibrator 초기화"""
        pass

    @abstractmethod
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """
        보정 모델 학습

        Args:
            y_prob: 예측 확률값
                   - Binary: shape (n_samples,) - positive class 확률
                   - Multi-class: shape (n_samples, n_classes) - 각 클래스별 확률
            y_true: 실제 레이블 (n_samples,)

        Returns:
            BaseCalibrator: 학습된 calibrator 객체 (method chaining 지원)

        Raises:
            ValueError: 입력 데이터가 잘못된 경우
        """
        pass

    @abstractmethod
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        확률값 보정

        Args:
            y_prob: 보정할 예측 확률값
                   - Binary: shape (n_samples,) - positive class 확률
                   - Multi-class: shape (n_samples, n_classes) - 각 클래스별 확률

        Returns:
            np.ndarray: 보정된 확률값 (입력과 같은 shape)

        Raises:
            ValueError: fit()이 호출되지 않았거나 입력 데이터가 잘못된 경우
        """
        pass

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        학습 후 변환 (편의 메서드)

        Args:
            y_prob: 예측 확률값
            y_true: 실제 레이블

        Returns:
            np.ndarray: 보정된 확률값
        """
        return self.fit(y_prob, y_true).transform(y_prob)

    @property
    @abstractmethod
    def supports_multiclass(self) -> bool:
        """
        다중 클래스 지원 여부

        Returns:
            bool: True if multi-class classification is supported
        """
        pass

    def __getstate__(self):
        """MLflow 직렬화 지원"""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """MLflow 역직렬화 지원"""
        self.__dict__.update(state)
