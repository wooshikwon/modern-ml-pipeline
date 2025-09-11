"""
Optuna 통합 관련 유틸리티 모음
"""
from typing import Dict, Any, Optional
from src.utils.system.logger import logger


def _require_optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except Exception as e:
        raise ImportError("Optuna가 설치되지 않았습니다. hyperparameter tuning을 사용하려면 optuna를 설치하세요.") from e


def logging_callback(study, trial):
    """
    Optuna 스터디의 각 trial이 완료될 때마다 진행 상황을 로깅하는 콜백 함수.
    Pruned된 trial도 안전하게 처리합니다.
    """
    try:
        current_value = getattr(trial, "value", None)
        best_value = getattr(study, "best_value", None)
        value_log = f"{current_value:.5f}" if current_value is not None else "N/A (pruned)"
        best_value_log = f"{best_value:.5f}" if best_value is not None else "N/A"
        logger.info(
            f"Trial {getattr(trial, 'number', 'N/A')} 완료 | "
            f"현재 점수: {value_log} | "
            f"최고 점수: {best_value_log}"
        )
    except Exception:
        logger.info("Optuna 로깅 콜백에서 정보를 읽을 수 없습니다.")


class OptunaIntegration:
    """Optuna 연동 도우미 클래스"""
    def __init__(self, tuning_config: Any, seed: Optional[int] = None, timeout: Optional[int] = None, n_jobs: Optional[int] = None, pruning: Optional[Dict[str, Any]] = None):
        self.tuning_config = tuning_config
        self.seed = seed
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.pruning = pruning or {}

    def create_study(self, direction: str, study_name: str, pruner=None):
        optuna = _require_optuna()
        sampler = optuna.samplers.TPESampler(seed=self.seed) if self.seed is not None else optuna.samplers.TPESampler()
        study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler, pruner=pruner)
        return study

    def suggest_hyperparameters(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        recipe.model.hyperparameters.root dict를 읽고, Optuna 파라미터 명세(type/low/high/choices 등)는 suggest_*,
        그 외는 고정값으로 그대로 반환.
        """
        suggested: Dict[str, Any] = {}
        for key, value in param_space.items():
            if isinstance(value, dict) and 'type' in value:
                ptype = value.get('type')
                _require_optuna()
                if ptype == 'int':
                    suggested[key] = trial.suggest_int(key, int(value['low']), int(value['high']), log=bool(value.get('log', False)))
                elif ptype == 'float':
                    suggested[key] = trial.suggest_float(key, float(value['low']), float(value['high']), log=bool(value.get('log', False)))
                elif ptype == 'categorical':
                    suggested[key] = trial.suggest_categorical(key, list(value.get('choices', [])))
                else:
                    suggested[key] = value
            else:
                suggested[key] = value
        return suggested 