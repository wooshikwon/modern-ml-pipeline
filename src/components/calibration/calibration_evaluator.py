"""
Calibration Evaluator - 확률 보정 평가 모듈

Expected Calibration Error (ECE) 등 캘리브레이션 성능 평가 메트릭 계산 및
Binary/Multiclass 분기 처리를 담당한다.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import brier_score_loss

from src.utils.core.logger import logger


class CalibrationEvaluator:
    """
    Calibration 평가 로직을 캡슐화한 클래스.
    모델과 calibrator를 받아 Binary/Multiclass 분기 처리 및 평가를 수행한다.
    """

    def __init__(self, trained_model, trained_calibrator):
        """
        CalibrationEvaluator 초기화

        Args:
            trained_model: 학습된 모델 (predict_proba 메서드 필요)
            trained_calibrator: 학습된 calibrator (transform 메서드 필요)
        """
        self.trained_model = trained_model
        self.trained_calibrator = trained_calibrator

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Calibration 평가 수행.

        Args:
            X_test: 테스트 특성
            y_test: 테스트 라벨

        Returns:
            Calibration metrics 딕셔너리 (flat 구조, MLflow 로깅용)
        """
        # Uncalibrated 확률 얻기
        y_prob_uncalibrated = self.trained_model.predict_proba(X_test)
        y_prob_calibrated = self.trained_calibrator.transform(y_prob_uncalibrated)

        # Binary vs Multiclass 자동 구분 및 평가
        if y_prob_uncalibrated.ndim == 2 and y_prob_uncalibrated.shape[1] == 2:
            # Binary classification - positive class 확률만 사용
            calibration_metrics = self._evaluate_binary(
                y_test,
                y_prob_uncalibrated[:, 1],
                y_prob_calibrated[:, 1] if y_prob_calibrated.ndim == 2 else y_prob_calibrated,
            )
            logger.info("[EVAL:Calibration] Binary 평가 완료")

        elif y_prob_uncalibrated.ndim == 2 and y_prob_uncalibrated.shape[1] > 2:
            # Multiclass classification
            calibration_metrics = self._evaluate_multiclass(
                y_test, y_prob_uncalibrated, y_prob_calibrated
            )
            logger.info(
                f"[EVAL:Calibration] Multiclass 평가 완료 ({y_prob_uncalibrated.shape[1]}클래스)"
            )

        else:
            # 1D case (이미 binary positive class만 있는 경우)
            calibration_metrics = self._evaluate_binary(
                y_test, y_prob_uncalibrated, y_prob_calibrated
            )
            logger.info("[EVAL:Calibration] 1D 평가 완료")

        # Nested dict 제거 (MLflow 로깅을 위해)
        flat_metrics = {}
        for key, value in calibration_metrics.items():
            if not isinstance(value, dict):  # class_metrics 같은 nested dict 제외
                flat_metrics[f"calibration_{key}"] = value

        return flat_metrics

    def _evaluate_binary(
        self,
        y_true: np.ndarray,
        y_prob_uncalibrated: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        이진 분류 캘리브레이션 전후 성능 비교 평가

        Args:
            y_true: 실제 라벨 (0 또는 1)
            y_prob_uncalibrated: 캘리브레이션 전 예측 확률
            y_prob_calibrated: 캘리브레이션 후 예측 확률 (선택)
            n_bins: ECE/MCE 계산용 구간 수

        Returns:
            캘리브레이션 메트릭 딕셔너리
        """
        y_true = np.asarray(y_true)
        y_prob_uncalibrated = np.asarray(y_prob_uncalibrated)

        if len(y_true) != len(y_prob_uncalibrated):
            raise ValueError("y_true and y_prob_uncalibrated must have the same length")

        if y_prob_calibrated is not None and len(y_true) != len(y_prob_calibrated):
            raise ValueError("y_true and y_prob_calibrated must have the same length")

        if not np.all((y_true == 0) | (y_true == 1)):
            raise ValueError("y_true must contain only binary values (0, 1)")

        metrics = {}

        # Uncalibrated metrics
        metrics["ece_uncalibrated"] = self._expected_calibration_error(
            y_true, y_prob_uncalibrated, n_bins
        )
        metrics["mce_uncalibrated"] = self._maximum_calibration_error(
            y_true, y_prob_uncalibrated, n_bins
        )
        metrics["brier_score_uncalibrated"] = brier_score_loss(y_true, y_prob_uncalibrated)

        # Calibrated metrics (if available)
        if y_prob_calibrated is not None:
            metrics["ece_calibrated"] = self._expected_calibration_error(
                y_true, y_prob_calibrated, n_bins
            )
            metrics["mce_calibrated"] = self._maximum_calibration_error(
                y_true, y_prob_calibrated, n_bins
            )
            metrics["brier_score_calibrated"] = brier_score_loss(y_true, y_prob_calibrated)

            # Improvement metrics
            metrics["ece_improvement"] = metrics["ece_uncalibrated"] - metrics["ece_calibrated"]
            metrics["ece_improvement_ratio"] = (
                (metrics["ece_improvement"] / metrics["ece_uncalibrated"])
                if metrics["ece_uncalibrated"] > 0
                else 0.0
            )
            metrics["mce_improvement"] = metrics["mce_uncalibrated"] - metrics["mce_calibrated"]
            metrics["brier_improvement"] = (
                metrics["brier_score_uncalibrated"] - metrics["brier_score_calibrated"]
            )

            logger.info(
                f"[EVAL:Calibration] ECE 개선: {metrics['ece_improvement']:.4f} "
                f"({metrics['ece_improvement_ratio']:.1%})"
            )

        return metrics

    def _evaluate_multiclass(
        self,
        y_true: np.ndarray,
        y_prob_uncalibrated: np.ndarray,
        y_prob_calibrated: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        다중 클래스 캘리브레이션 평가 (One-vs-Rest 방식)

        Args:
            y_true: 실제 라벨 (클래스 인덱스)
            y_prob_uncalibrated: 캘리브레이션 전 예측 확률 (n_samples, n_classes)
            y_prob_calibrated: 캘리브레이션 후 예측 확률 (n_samples, n_classes)
            n_bins: 구간 수

        Returns:
            클래스별 및 평균 캘리브레이션 메트릭
        """
        n_classes = y_prob_uncalibrated.shape[1]
        metrics = {"n_classes": n_classes}

        class_ece_uncal = []
        class_mce_uncal = []
        class_ece_cal = []
        class_mce_cal = []

        for class_idx in range(n_classes):
            # 이진 분류 문제로 변환 (One-vs-Rest)
            y_binary = (y_true == class_idx).astype(int)
            y_prob_class_uncal = y_prob_uncalibrated[:, class_idx]
            y_prob_class_cal = (
                y_prob_calibrated[:, class_idx] if y_prob_calibrated is not None else None
            )

            ece_uncal = self._expected_calibration_error(y_binary, y_prob_class_uncal, n_bins)
            mce_uncal = self._maximum_calibration_error(y_binary, y_prob_class_uncal, n_bins)

            class_ece_uncal.append(ece_uncal)
            class_mce_uncal.append(mce_uncal)

            if y_prob_class_cal is not None:
                class_ece_cal.append(
                    self._expected_calibration_error(y_binary, y_prob_class_cal, n_bins)
                )
                class_mce_cal.append(
                    self._maximum_calibration_error(y_binary, y_prob_class_cal, n_bins)
                )

        # 클래스별 메트릭의 평균 계산
        metrics["avg_ece_uncalibrated"] = np.mean(class_ece_uncal)
        metrics["avg_mce_uncalibrated"] = np.mean(class_mce_uncal)

        if y_prob_calibrated is not None:
            metrics["avg_ece_calibrated"] = np.mean(class_ece_cal)
            metrics["avg_mce_calibrated"] = np.mean(class_mce_cal)
            metrics["avg_ece_improvement"] = (
                metrics["avg_ece_uncalibrated"] - metrics["avg_ece_calibrated"]
            )
            metrics["avg_ece_improvement_ratio"] = (
                (metrics["avg_ece_improvement"] / metrics["avg_ece_uncalibrated"])
                if metrics["avg_ece_uncalibrated"] > 0
                else 0.0
            )

        return metrics

    def _expected_calibration_error(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) 계산

        ECE는 모델의 확률 예측이 얼마나 잘 보정되어 있는지를 측정한다.

        Args:
            y_true: 실제 라벨 (0 또는 1)
            y_prob: 예측 확률 (0~1 사이)
            n_bins: 구간 수 (기본값: 10)

        Returns:
            ECE 값 (0에 가까울수록 잘 보정된 모델)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _maximum_calibration_error(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Maximum Calibration Error (MCE) 계산

        Args:
            y_true: 실제 라벨 (0 또는 1)
            y_prob: 예측 확률 (0~1 사이)
            n_bins: 구간 수 (기본값: 10)

        Returns:
            MCE 값 (최대 보정 오차)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_errors = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_errors.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return max(calibration_errors) if calibration_errors else 0.0


# 하위 호환성을 위한 별칭
CalibrationEvaluatorWrapper = CalibrationEvaluator
