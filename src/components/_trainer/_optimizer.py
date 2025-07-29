# src/components/_trainer/_optimizer.py
from datetime import datetime
from typing import Dict, Any, Callable
import pandas as pd
from src.settings import Settings
from src.utils.system.logger import logger

class OptunaOptimizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def optimize(self, train_df: pd.DataFrame, training_callback: Callable) -> Dict[str, Any]:
        """Optuna를 사용하여 하이퍼파라미터 최적화를 수행합니다."""
        logger.info("🚀 하이퍼파라미터 자동 최적화 모드 시작")
        
        from src.engine import Factory
        factory = Factory(self.settings)
        optuna_integration = factory.create_optuna_integration()

        study = optuna_integration.create_study(
            direction=self.settings.recipe.model.hyperparameter_tuning.direction,
            study_name=f"study_{self.settings.recipe.model.computed['run_name']}"
        )
        
        start_time = datetime.now()

        def objective(trial):
            # optuna_integration을 통해 파라미터 제안
            params = optuna_integration.suggest_hyperparameters(
                trial, self.settings.recipe.model.hyperparameters.root
            )
            result = training_callback(train_df=train_df, params=params, seed=trial.number)
            return result['score']
        
        study.optimize(
            objective, 
            n_trials=self.settings.recipe.model.hyperparameter_tuning.n_trials,
            timeout=getattr(self.settings.hyperparameter_tuning, 'timeout', None)
        )
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 하이퍼파라미터 최적화 완료! 최고 점수: {study.best_value:.4f} ({optimization_time:.1f}초)")

        return {
            'enabled': True,
            'engine': 'optuna',
            'best_params': study.best_params,
            'best_score': study.best_value,
            'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state.name == 'PRUNED']),
            'optimization_time': optimization_time,
            'search_space': {k: v for k, v in self.settings.recipe.model.hyperparameters.root.items() if isinstance(v, dict)}
        }
