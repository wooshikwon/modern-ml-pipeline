"""
Optuna 통합 관련 유틸리티 모음
"""
import optuna
from src.utils.system.logger import logger

def logging_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Optuna 스터디의 각 trial이 완료될 때마다 진행 상황을 로깅하는 콜백 함수.
    Pruned된 trial도 안전하게 처리합니다.
    """
    current_value = trial.value
    best_value = study.best_value
    
    value_log = f"{current_value:.5f}" if current_value is not None else "N/A (pruned)"
    best_value_log = f"{best_value:.5f}" if best_value is not None else "N/A"
    
    logger.info(
        f"Trial {trial.number} 완료 | "
        f"현재 점수: {value_log} | "
        f"최고 점수: {best_value_log}"
    ) 