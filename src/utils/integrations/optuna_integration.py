from __future__ import annotations

import importlib.util


def require_optuna() -> None:
    if importlib.util.find_spec("optuna") is None:
        # 테스트 기대 메시지와 일치
        raise ImportError(
            "Optuna가 설치되지 않았습니다. hyperparameter tuning을 사용하려면 optuna를 설치하세요."
        )


"""
Optuna 통합 관련 유틸리티 모음
"""
from typing import Any, Dict, Optional

from src.utils.core.logger import logger


def _require_optuna():
    try:
        import optuna as _optuna  # type: ignore

        # Optuna 내부 로그 억제 (우리 형식의 콜백으로 대체)
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        return _optuna
    except Exception as e:
        raise ImportError(
            "Optuna가 설치되지 않았습니다. hyperparameter tuning을 사용하려면 optuna를 설치하세요."
        ) from e


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
            f"[TRAIN:HPO] Trial {getattr(trial, 'number', 'N/A')} 완료 | "
            f"현재: {value_log} | 최고: {best_value_log}"
        )
    except Exception:
        logger.warning("[TRAIN:HPO] Optuna 콜백에서 정보를 읽을 수 없습니다.")


class OptunaIntegration:
    """Optuna 연동 도우미 클래스"""

    def __init__(
        self,
        tuning_config: Any,
        seed: Optional[int] = None,
        timeout: Optional[int] = None,
        n_jobs: Optional[int] = None,
        pruning: Optional[Dict[str, Any]] = None,
    ):
        self.tuning_config = tuning_config
        self.seed = seed
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.pruning = pruning or {}

    def create_study(self, direction: str, study_name: str, pruner=None):
        optuna = _require_optuna()
        sampler = (
            optuna.samplers.TPESampler(seed=self.seed)
            if self.seed is not None
            else optuna.samplers.TPESampler()
        )
        study = optuna.create_study(
            direction=direction, study_name=study_name, sampler=sampler, pruner=pruner
        )
        return study

    def suggest_hyperparameters(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        recipe.model.hyperparameters.root dict를 읽고, Optuna 파라미터 명세(type/low/high/choices 등)는 suggest_*,
        그 외는 고정값으로 그대로 반환.
        """
        suggested: Dict[str, Any] = {}
        for key, value in param_space.items():
            if isinstance(value, dict) and "type" in value:
                ptype = value.get("type")
                _require_optuna()
                # range: [min, max] 형식과 low/high 형식 모두 지원
                if "range" in value:
                    low, high = value["range"][0], value["range"][1]
                else:
                    low, high = value.get("low"), value.get("high")
                if ptype == "int":
                    suggested[key] = trial.suggest_int(
                        key, int(low), int(high), log=bool(value.get("log", False))
                    )
                elif ptype == "float":
                    suggested[key] = trial.suggest_float(
                        key, float(low), float(high), log=bool(value.get("log", False))
                    )
                elif ptype == "categorical":
                    suggested[key] = trial.suggest_categorical(key, list(value.get("choices", [])))
                else:
                    suggested[key] = value
            else:
                suggested[key] = value
        return suggested
