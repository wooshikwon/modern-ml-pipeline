from typing import Dict, Any, Optional
from src.settings.settings import HyperparameterTuningSettings
from src.utils.system.logger import logger

# Optuna는 선택적 의존성으로 처리
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False


class OptunaAdapter:
    """Optuna SDK 래퍼 클래스"""
    
    def __init__(self, hyperparameter_tuning_settings: HyperparameterTuningSettings):
        if not HAS_OPTUNA:
            raise ImportError("Optuna 라이브러리가 필요합니다. `pip install optuna`로 설치하세요.")
        
        self.settings = hyperparameter_tuning_settings
        logger.info("Optuna 어댑터가 초기화되었습니다.")
    
    def create_study(
        self,
        direction: str = "maximize", 
        pruner: Optional[Any] = None,
        study_name: Optional[str] = None
    ):
        """Optuna Study 생성"""
        if not HAS_OPTUNA:
            raise ImportError("Optuna가 설치되지 않았습니다.")
        
        # 기본 Pruner 설정
        if pruner is None and self.settings.pruning and self.settings.pruning.get("enabled", True):
            pruning_config = self.settings.pruning
            algorithm = pruning_config.get("algorithm", "MedianPruner")
            
            if algorithm == "MedianPruner":
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=pruning_config.get("n_startup_trials", 5),
                    n_warmup_steps=pruning_config.get("n_warmup_steps", 10)
                )
            else:
                logger.warning(f"지원하지 않는 Pruner 알고리즘: {algorithm}, MedianPruner 사용")
                pruner = optuna.pruners.MedianPruner()
        
        study = optuna.create_study(
            direction=direction,
            pruner=pruner,
            study_name=study_name
        )
        
        logger.info(f"Optuna Study 생성 완료 (direction: {direction}, pruner: {type(pruner).__name__})")
        return study
    
    def suggest_hyperparameter(self, trial, param_name: str, param_config: Dict[str, Any]):
        """단일 하이퍼파라미터 제안"""
        if not isinstance(param_config, dict) or 'type' not in param_config:
            # 고정값인 경우
            return param_config
        
        param_type = param_config['type']
        
        if param_type == 'float':
            low = param_config['low']
            high = param_config['high']
            log = param_config.get('log', False)
            return trial.suggest_float(param_name, low, high, log=log)
        elif param_type == 'int':
            low = param_config['low']
            high = param_config['high']
            return trial.suggest_int(param_name, low, high)
        elif param_type == 'categorical':
            choices = param_config['choices']
            return trial.suggest_categorical(param_name, choices)
        else:
            logger.warning(f"지원하지 않는 파라미터 타입: {param_type}, 고정값으로 처리")
            return param_config.get('default', None)
    
    def suggest_hyperparameters(self, trial, hyperparams_config: Dict[str, Any]) -> Dict[str, Any]:
        """전체 하이퍼파라미터 제안"""
        suggested_params = {}
        
        for param_name, param_config in hyperparams_config.items():
            suggested_params[param_name] = self.suggest_hyperparameter(
                trial, param_name, param_config
            )
        
        return suggested_params
    
    def is_available(self) -> bool:
        """Optuna 사용 가능 여부 확인"""
        return HAS_OPTUNA 