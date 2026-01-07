from typing import List, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.components.calibration.base import BaseCalibrator
from src.utils.core.logger import logger

from ..registry import CalibrationRegistry


class IsotonicCalibration(BaseCalibrator):
    """
    Isotonic Regression을 이용한 확률 보정 (Binary/Multi-class Classification)

    단조 회귀(Monotonic Regression)를 사용하여 분류 모델의 예측 확률을 보정합니다.
    비모수 방법으로 확률과 실제 레이블 간의 단조 관계를 학습합니다.

    Binary Classification: 직접 Isotonic Regression 적용
    Multi-class Classification: One-vs-Rest 방식으로 각 클래스별 보정

    References:
        Zadrozny and Elkan (2002). "Transforming classifier scores into accurate multiclass probability estimates"
    """

    def __init__(self):
        """IsotonicCalibration 초기화"""
        super().__init__()
        self.calibrator: Union[IsotonicRegression, List[IsotonicRegression], None] = None
        self._is_fitted = False
        self._supports_multiclass = True
        self._n_classes = None

        logger.info("[FACT:Calibrator] IsotonicCalibration 초기화 완료")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibration":
        """
        Isotonic Regression 모델 학습

        Args:
            y_prob: 예측 확률값
                   - Binary: shape (n_samples,) - positive class 확률
                   - Multi-class: shape (n_samples, n_classes) - 각 클래스별 확률
            y_true: 실제 레이블 (n_samples,)

        Returns:
            IsotonicCalibration: 학습된 calibrator 객체

        Raises:
            ValueError: 입력 형식이 잘못된 경우
        """
        logger.info(f"[TRAIN:Calibrator] Isotonic Calibration 학습 시작 - 샘플: {len(y_prob):,}개")

        # 입력 검증
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)

        logger.debug(
            f"[TRAIN:Calibrator] 입력 데이터 검증 - y_prob: {y_prob.shape}, y_true: {y_true.shape}"
        )

        if len(y_prob) != len(y_true):
            raise ValueError(f"y_prob와 y_true의 길이가 다릅니다: {len(y_prob)} vs {len(y_true)}")

        # 클래스 수 확인
        unique_labels = np.unique(y_true)
        self._n_classes = len(unique_labels)

        logger.debug(
            f"[TRAIN:Calibrator] 클래스 구조 분석 - {self._n_classes}개 클래스 발견: {unique_labels}"
        )

        if self._n_classes < 2:
            raise ValueError(f"최소 2개 클래스가 필요합니다. 발견된 클래스: {unique_labels}")

        if y_prob.ndim == 1:  # Binary Classification
            if self._n_classes > 2:
                raise ValueError(
                    f"이진 확률값(1D)이지만 {self._n_classes}개 클래스가 발견되었습니다."
                )

            # 확률값 범위 검증
            if not (0 <= y_prob.min() and y_prob.max() <= 1):
                raise ValueError(
                    f"확률값은 [0, 1] 범위에 있어야 합니다. 범위: [{y_prob.min():.4f}, {y_prob.max():.4f}]"
                )

            # Binary classification: 직접 Isotonic Regression 적용
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_true)

        elif y_prob.ndim == 2:  # Multi-class Classification
            if y_prob.shape[1] != self._n_classes:
                raise ValueError(
                    f"확률 행렬의 클래스 수({y_prob.shape[1]})와 실제 클래스 수({self._n_classes})가 다릅니다."
                )

            # 확률값 범위 검증
            if not (0 <= y_prob.min() and y_prob.max() <= 1):
                raise ValueError(
                    f"확률값은 [0, 1] 범위에 있어야 합니다. 범위: [{y_prob.min():.4f}, {y_prob.max():.4f}]"
                )

            # Multi-class: One-vs-Rest 방식으로 각 클래스별 보정
            self.calibrator = []
            for class_idx in range(self._n_classes):
                # 현재 클래스 vs 나머지로 이진 분류 문제 구성
                binary_labels = (y_true == unique_labels[class_idx]).astype(int)
                class_probs = y_prob[:, class_idx]

                # 해당 클래스의 Isotonic Regression 학습
                iso_reg = IsotonicRegression(out_of_bounds="clip")
                iso_reg.fit(class_probs, binary_labels)
                self.calibrator.append(iso_reg)

        else:
            raise ValueError(f"y_prob는 1D 또는 2D 배열이어야 합니다. 받은 shape: {y_prob.shape}")

        self._is_fitted = True

        # 학습 결과 요약
        if self._n_classes == 2:
            threshold_count = (
                len(self.calibrator.X_thresholds_)
                if hasattr(self.calibrator, "X_thresholds_")
                else 0
            )
            logger.info(
                f"[TRAIN:Calibrator] Isotonic Calibration 학습 완료 - 이진 분류, 임계값: {threshold_count}개"
            )
        else:
            total_thresholds = sum(
                len(cal.X_thresholds_) if hasattr(cal, "X_thresholds_") else 0
                for cal in self.calibrator
            )
            logger.info(
                f"[TRAIN:Calibrator] Isotonic Calibration 학습 완료 - Multi-class, 임계값: {total_thresholds}개 ({self._n_classes}클래스)"
            )

        return self

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
            ValueError: fit()이 호출되지 않았거나 입력 형식이 잘못된 경우
        """
        if not self._is_fitted:
            raise ValueError("transform()을 호출하기 전에 fit()을 먼저 호출해야 합니다.")

        # 입력 검증
        y_prob = np.asarray(y_prob)

        if y_prob.ndim == 1:  # Binary Classification
            if not isinstance(self.calibrator, IsotonicRegression):
                raise ValueError("이진 확률값이지만 다중 클래스 calibrator로 학습되었습니다.")

            # 확률값 범위 검증
            if not (0 <= y_prob.min() and y_prob.max() <= 1):
                raise ValueError(
                    f"확률값은 [0, 1] 범위에 있어야 합니다. 범위: [{y_prob.min():.4f}, {y_prob.max():.4f}]"
                )

            # Isotonic Regression 적용
            calibrated_probs = self.calibrator.transform(y_prob)
            return calibrated_probs

        elif y_prob.ndim == 2:  # Multi-class Classification
            if not isinstance(self.calibrator, list):
                raise ValueError("다중 클래스 확률값이지만 이진 분류 calibrator로 학습되었습니다.")

            if y_prob.shape[1] != len(self.calibrator):
                raise ValueError(
                    f"확률 행렬의 클래스 수({y_prob.shape[1]})와 학습된 클래스 수({len(self.calibrator)})가 다릅니다."
                )

            # 확률값 범위 검증
            if not (0 <= y_prob.min() and y_prob.max() <= 1):
                raise ValueError(
                    f"확률값은 [0, 1] 범위에 있어야 합니다. 범위: [{y_prob.min():.4f}, {y_prob.max():.4f}]"
                )

            # 각 클래스별로 보정 적용
            calibrated_probs = np.zeros_like(y_prob)
            for class_idx, iso_reg in enumerate(self.calibrator):
                calibrated_probs[:, class_idx] = iso_reg.transform(y_prob[:, class_idx])

            # 확률 정규화 (합이 1이 되도록)
            # 각 행(샘플)에 대해 정규화
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            # 0으로 나누기 방지
            row_sums = np.where(row_sums == 0, 1, row_sums)
            calibrated_probs = calibrated_probs / row_sums

            return calibrated_probs

        else:
            raise ValueError(f"y_prob는 1D 또는 2D 배열이어야 합니다. 받은 shape: {y_prob.shape}")

    @property
    def supports_multiclass(self) -> bool:
        """
        다중 클래스 지원 여부

        Returns:
            bool: True (Isotonic Regression은 다중 클래스 지원)
        """
        return self._supports_multiclass

    def __getstate__(self):
        """MLflow 직렬화 지원"""
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        """MLflow 역직렬화 지원"""
        super().__setstate__(state)


# 자동 등록
CalibrationRegistry.register("isotonic", IsotonicCalibration)
