"""
Beta Calibration 모듈
Beta 분포를 사용한 확률 보정 방법
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from src.components.calibration.base import BaseCalibrator
from src.utils.core.logger import logger

from ..registry import CalibrationRegistry


class BetaCalibration(BaseCalibrator):
    """
    Beta Calibration을 사용한 확률 보정

    Beta 분포의 파라미터를 추정하여 확률을 보정합니다.
    이진 분류와 다중 분류 모두 지원합니다.

    References:
        Kull, M., Silva Filho, T. M., & Flach, P. (2017).
        Beta calibration: a well-founded and easily implemented improvement on Platt scaling
        for binary and multiclass classification.
        Proceedings of the 20th International Conference on Artificial Intelligence and Statistics.
    """

    def __init__(self):
        logger.info("[BetaCalibration] 초기화 시작합니다")

        super().__init__()
        self.parameters = None  # Beta 분포의 파라미터들
        self._n_classes = None
        self._is_fitted = False

        logger.info("[BetaCalibration] 초기화 완료되었습니다")

    @property
    def supports_multiclass(self) -> bool:
        return True

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BetaCalibration":
        """
        Beta Calibration 파라미터 학습

        Args:
            y_prob: 예측 확률 (1D for binary, 2D for multiclass)
            y_true: 실제 라벨

        Returns:
            학습된 BetaCalibration 객체
        """
        logger.info(f"Beta Calibration 학습 시작 - 데이터 샘플: {len(y_prob):,}개")

        # 입력 검증
        y_prob = np.asarray(y_prob)
        y_true = np.asarray(y_true)

        if len(y_prob) != len(y_true):
            logger.error("[BetaCalibration] 입력 데이터 길이 불일치")
            raise ValueError("y_prob와 y_true의 길이가 다릅니다")

        # 확률값 범위 검증
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            logger.error("[BetaCalibration] 확률값 범위 오류: [0,1] 범위를 벗어남")
            raise ValueError("확률값은 [0, 1] 범위에 있어야 합니다")

        # 클래스 수 결정
        unique_labels = np.unique(y_true)
        self._n_classes = len(unique_labels)

        if self._n_classes < 2:
            logger.error(f"[BetaCalibration] 클래스 수 부족: {self._n_classes}개 (최소 2개 필요)")
            raise ValueError("최소 2개 클래스가 필요합니다")

        # 이진 분류 vs 다중 분류
        if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
            # 이진 분류
            if self._n_classes > 2:
                logger.error(
                    f"[BetaCalibration] 분류 유형 불일치: 이진 확률(1D)이지만 {self._n_classes}개 클래스 발견"
                )
                raise ValueError(
                    f"이진 확률값(1D)이지만 {self._n_classes}개 클래스가 발견되었습니다"
                )

            logger.info(f"Binary Classification Beta 파라미터 추정 - 데이터 형태: {y_prob.shape}")
            y_prob_flat = y_prob.flatten() if y_prob.ndim == 2 else y_prob
            self.parameters = self._fit_beta_parameters(y_prob_flat, y_true)

        elif y_prob.ndim == 2:
            # 다중 분류
            if y_prob.shape[1] != self._n_classes:
                logger.error(
                    f"[BetaCalibration] 클래스 수 불일치: 확률 행렬({y_prob.shape[1]}) vs 실제({self._n_classes})"
                )
                raise ValueError(
                    f"확률 행렬의 클래스 수({y_prob.shape[1]})와 실제 클래스 수({self._n_classes})가 다릅니다"
                )

            logger.info(
                f"Multiclass One-vs-Rest Beta Calibration - {self._n_classes}개 클래스 각각 파라미터 추정"
            )

            # 각 클래스별로 One-vs-Rest Beta calibration
            self.parameters = []
            for class_idx in range(self._n_classes):
                # 이진 라벨 생성 (현재 클래스 vs 나머지)
                binary_labels = (y_true == unique_labels[class_idx]).astype(int)
                class_probs = y_prob[:, class_idx]

                # 클래스별 Beta 파라미터 추정
                params = self._fit_beta_parameters(class_probs, binary_labels)
                self.parameters.append(params)
                logger.debug(
                    f"Class {class_idx} Beta 파라미터 추정 완료 - a={params['a']:.4f}, b={params['b']:.4f}"
                )
        else:
            logger.error("[BetaCalibration] 데이터 차원 오류: y_prob는 1D 또는 2D 배열이어야 함")
            raise ValueError("y_prob는 1D 또는 2D 배열이어야 합니다")

        self._is_fitted = True

        logger.info(f"Beta Calibration 학습 완료 - {self._n_classes}개 클래스, 파라미터 추정 성공")
        return self

    def _fit_beta_parameters(self, y_prob: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Beta 분포 파라미터 추정

        Args:
            y_prob: 예측 확률 (1D)
            y_true: 이진 라벨 (0 또는 1)

        Returns:
            추정된 파라미터들
        """
        # 경계값 처리 (0과 1을 피함)
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

        # Logit 변환
        logit_probs = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # 목적 함수 정의 (음의 로그 우도)
        def neg_log_likelihood(params):
            a, b = params

            # Beta 분포의 평균과 분산 계산
            # E[X] = a / (a + b)
            # Var[X] = ab / ((a+b)^2 * (a+b+1))

            if a <= 0 or b <= 0:
                return np.inf

            # Calibrated 확률 계산
            # P_calibrated = sigmoid(a * logit(p) + b)
            calibrated_logits = a * logit_probs + b
            calibrated_probs = expit(calibrated_logits)

            # 로그 우도 계산
            calibrated_probs = np.clip(calibrated_probs, epsilon, 1 - epsilon)
            log_likelihood = np.sum(
                y_true * np.log(calibrated_probs) + (1 - y_true) * np.log(1 - calibrated_probs)
            )

            return -log_likelihood

        # 초기값 설정
        initial_params = [1.0, 0.0]  # a=1, b=0 (항등 변환)

        # 최적화
        try:
            result = minimize(
                neg_log_likelihood,
                initial_params,
                method="L-BFGS-B",
                bounds=[(0.01, 100), (-10, 10)],  # a > 0, b 제한
            )

            if result.success:
                a_opt, b_opt = result.x
            else:
                # 최적화 실패 시 기본값
                a_opt, b_opt = 1.0, 0.0

        except Exception:
            # 오류 발생 시 기본값
            a_opt, b_opt = 1.0, 0.0

        return {"a": a_opt, "b": b_opt}

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        확률 보정 변환 수행

        Args:
            y_prob: 원본 예측 확률

        Returns:
            보정된 확률
        """
        if not self._is_fitted:
            logger.error("[BetaCalibration] 모델이 학습되지 않음: fit() 먼저 호출 필요")
            raise ValueError("fit()을 먼저 호출해야 합니다")

        logger.info(
            f"Beta Calibration 변환 시작 - 원본 확률: {len(y_prob):,}개 샘플, 형태: {y_prob.shape}"
        )

        y_prob = np.asarray(y_prob)

        # 이진 분류 vs 다중 분류 구분
        if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
            # 이진 분류
            if isinstance(self.parameters, list):
                logger.error(
                    "[BetaCalibration] 모델-데이터 불일치: 이진 확률이지만 다중분류 calibrator 사용"
                )
                raise ValueError("다중 클래스 확률값이지만 이진 분류 calibrator로 학습되었습니다")

            y_prob_flat = y_prob.flatten() if y_prob.ndim == 2 else y_prob
            calibrated = self._apply_beta_calibration(y_prob_flat, self.parameters)

            return calibrated.reshape(y_prob.shape) if y_prob.ndim == 2 else calibrated

        elif y_prob.ndim == 2:
            # 다중 분류
            if not isinstance(self.parameters, list):
                logger.error(
                    "[BetaCalibration] 모델-데이터 불일치: 다중분류 확률이지만 이진분류 calibrator 사용"
                )
                raise ValueError("이진 확률값이지만 다중 클래스 calibrator로 학습되었습니다")

            if y_prob.shape[1] != len(self.parameters):
                logger.error(
                    f"[BetaCalibration] 클래스 수 불일치: 입력({y_prob.shape[1]}) vs 학습된 모델({len(self.parameters)})"
                )
                raise ValueError(
                    f"확률 행렬의 클래스 수({y_prob.shape[1]})와 학습된 클래스 수({len(self.parameters)})가 다릅니다"
                )

            # 각 클래스별로 보정 적용
            calibrated_probs = np.zeros_like(y_prob)
            for class_idx, params in enumerate(self.parameters):
                calibrated_probs[:, class_idx] = self._apply_beta_calibration(
                    y_prob[:, class_idx], params
                )

            # 확률 정규화 (합이 1이 되도록)
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
            calibrated_probs = calibrated_probs / row_sums

            logger.info(
                f"Beta Calibration 변환 완료 - Multiclass 보정된 확률: {self._n_classes}개 클래스, 형태: {calibrated_probs.shape}"
            )
            return calibrated_probs
        else:
            logger.error("[BetaCalibration] 데이터 차원 오류: y_prob는 1D 또는 2D 배열이어야 함")
            raise ValueError("y_prob는 1D 또는 2D 배열이어야 합니다")

    def _apply_beta_calibration(self, y_prob: np.ndarray, params: dict) -> np.ndarray:
        """
        Beta calibration 변환 적용

        Args:
            y_prob: 예측 확률 (1D)
            params: Beta 파라미터

        Returns:
            보정된 확률
        """
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

        # Logit 변환
        logit_probs = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Beta calibration 적용: P_cal = sigmoid(a * logit(p) + b)
        a, b = params["a"], params["b"]
        calibrated_logits = a * logit_probs + b
        calibrated_probs = expit(calibrated_logits)

        return calibrated_probs

    def __getstate__(self):
        """MLflow 직렬화를 위한 상태 저장"""
        return {
            "parameters": self.parameters,
            "_n_classes": self._n_classes,
            "_is_fitted": self._is_fitted,
        }

    def __setstate__(self, state):
        """MLflow 역직렬화를 위한 상태 복원"""
        self.parameters = state["parameters"]
        self._n_classes = state["_n_classes"]
        self._is_fitted = state["_is_fitted"]


# Self-registration
CalibrationRegistry.register("beta", BetaCalibration)
