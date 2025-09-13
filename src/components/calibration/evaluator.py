"""
Calibration Evaluation 모듈
Expected Calibration Error (ECE) 등 캘리브레이션 성능 평가 메트릭
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import brier_score_loss

from src.utils.core.console_manager import get_console


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) 계산
    
    ECE는 모델의 확률 예측이 얼마나 잘 보정되어 있는지를 측정합니다.
    
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
        # 현재 구간에 속하는 샘플들
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # 구간 내 평균 정확도
            accuracy_in_bin = y_true[in_bin].mean()
            # 구간 내 평균 예측 확률
            avg_confidence_in_bin = y_prob[in_bin].mean()
            # ECE에 기여하는 부분
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Reliability Diagram을 위한 데이터 생성
    
    Args:
        y_true: 실제 라벨 (0 또는 1)
        y_prob: 예측 확률 (0~1 사이)
        n_bins: 구간 수 (기본값: 10)
        
    Returns:
        bin_centers, bin_accuracies, bin_confidences, bin_counts를 포함한 딕셔너리
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_count = in_bin.sum()
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_prob[in_bin].mean()
        else:
            bin_accuracy = 0.0
            bin_confidence = 0.0
        
        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
    
    return {
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


def evaluate_calibration_metrics(
    y_true: np.ndarray, 
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    캘리브레이션 전후 성능 비교 평가
    
    Args:
        y_true: 실제 라벨 (0 또는 1)
        y_prob_uncalibrated: 캘리브레이션 전 예측 확률
        y_prob_calibrated: 캘리브레이션 후 예측 확률 (선택)
        n_bins: ECE/MCE 계산용 구간 수
        
    Returns:
        캘리브레이션 메트릭 딕셔너리
    """
    # 입력 검증
    if len(y_true) != len(y_prob_uncalibrated):
        raise ValueError("y_true and y_prob_uncalibrated must have the same length")
    
    if y_prob_calibrated is not None and len(y_true) != len(y_prob_calibrated):
        raise ValueError("y_true and y_prob_calibrated must have the same length")
    
    # 이진 분류 확인 
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only binary values (0, 1)")
    
    metrics = {}
    
    # Uncalibrated metrics
    metrics['ece_uncalibrated'] = expected_calibration_error(y_true, y_prob_uncalibrated, n_bins)
    metrics['mce_uncalibrated'] = maximum_calibration_error(y_true, y_prob_uncalibrated, n_bins)
    metrics['brier_score_uncalibrated'] = brier_score_loss(y_true, y_prob_uncalibrated)
    
    # Calibrated metrics (if available)
    if y_prob_calibrated is not None:
        metrics['ece_calibrated'] = expected_calibration_error(y_true, y_prob_calibrated, n_bins)
        metrics['mce_calibrated'] = maximum_calibration_error(y_true, y_prob_calibrated, n_bins)
        metrics['brier_score_calibrated'] = brier_score_loss(y_true, y_prob_calibrated)
        
        # Improvement metrics
        metrics['ece_improvement'] = metrics['ece_uncalibrated'] - metrics['ece_calibrated']
        metrics['ece_improvement_ratio'] = (metrics['ece_improvement'] / metrics['ece_uncalibrated']) if metrics['ece_uncalibrated'] > 0 else 0.0
        metrics['mce_improvement'] = metrics['mce_uncalibrated'] - metrics['mce_calibrated']
        metrics['brier_improvement'] = metrics['brier_score_uncalibrated'] - metrics['brier_score_calibrated']
        
        console = get_console()
        console.log_model_operation(
            "Calibration 평가 완료",
            f"ECE 개선: {metrics['ece_improvement']:.4f} ({metrics['ece_improvement_ratio']:.1%})"
        )
    
    return metrics


def evaluate_multiclass_calibration(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10
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
    metrics = {'n_classes': n_classes}
    
    class_metrics = {}
    
    for class_idx in range(n_classes):
        # 이진 분류 문제로 변환 (One-vs-Rest)
        y_binary = (y_true == class_idx).astype(int)
        y_prob_class_uncal = y_prob_uncalibrated[:, class_idx]
        
        class_metrics_uncal = evaluate_calibration_metrics(
            y_binary, y_prob_class_uncal, 
            y_prob_calibrated[:, class_idx] if y_prob_calibrated is not None else None,
            n_bins
        )
        
        class_metrics[f'class_{class_idx}'] = class_metrics_uncal
    
    # 클래스별 메트릭의 평균 계산
    avg_ece_uncal = np.mean([class_metrics[f'class_{i}']['ece_uncalibrated'] for i in range(n_classes)])
    avg_mce_uncal = np.mean([class_metrics[f'class_{i}']['mce_uncalibrated'] for i in range(n_classes)])
    
    metrics['avg_ece_uncalibrated'] = avg_ece_uncal
    metrics['avg_mce_uncalibrated'] = avg_mce_uncal
    
    if y_prob_calibrated is not None:
        avg_ece_cal = np.mean([class_metrics[f'class_{i}']['ece_calibrated'] for i in range(n_classes)])
        avg_mce_cal = np.mean([class_metrics[f'class_{i}']['mce_calibrated'] for i in range(n_classes)])
        
        metrics['avg_ece_calibrated'] = avg_ece_cal
        metrics['avg_mce_calibrated'] = avg_mce_cal
        metrics['avg_ece_improvement'] = avg_ece_uncal - avg_ece_cal
        metrics['avg_ece_improvement_ratio'] = (metrics['avg_ece_improvement'] / avg_ece_uncal) if avg_ece_uncal > 0 else 0.0
    
    metrics['class_metrics'] = class_metrics
    
    return metrics