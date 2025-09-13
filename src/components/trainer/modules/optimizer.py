# src/components/_trainer/_optimizer.py
from datetime import datetime
from typing import Dict, Any, Callable
import pandas as pd
from src.settings import Settings
from src.utils.core.console_manager import get_console
from src.utils.integrations.optuna_integration import logging_callback
from ..registry import TrainerRegistry

class OptunaOptimizer:
    def __init__(self, settings: Settings, factory_provider: Callable[[], Any]):
        self.settings = settings
        self.factory_provider = factory_provider
        self.console = get_console(settings)
        self.console.info("[OptunaOptimizer] 초기화를 시작합니다")
        self.pruner = self._create_pruner()
        self.console.component_init("OptunaOptimizer", "success")
        self.console.info("[OptunaOptimizer] 초기화가 완료되었습니다")

    def _create_pruner(self):
        """기본 Pruner 생성 (필요시 settings로 확장 가능)"""
        try:
            import optuna
            return optuna.pruners.MedianPruner()
        except Exception:
            return None

    def optimize(self, train_df: pd.DataFrame, training_callback: Callable) -> Dict[str, Any]:
        """Optuna를 사용하여 하이퍼파라미터 최적화를 수행합니다."""
        n_trials = self.settings.recipe.model.hyperparameters.n_trials or 10
        self.console.info(f"[OptunaOptimizer] 하이퍼파라미터 최적화를 시작합니다 ({n_trials}회 시행)")
        
        factory = self.factory_provider()
        optuna_integration = factory.create_optuna_integration()

        # optimization_metric에 따라 direction 자동 결정
        optimization_metric = self.settings.recipe.model.hyperparameters.optimization_metric or "accuracy"
        
        # metric별 방향 매핑
        metric_directions = {
            # Classification - 모두 maximize
            "accuracy": "maximize", "precision": "maximize", "recall": "maximize", "f1": "maximize", "roc_auc": "maximize",
            # Regression - MSE, RMSE, MAE, MAPE는 minimize, R2는 maximize
            "mae": "minimize", "mse": "minimize", "rmse": "minimize", "r2": "maximize", "mape": "minimize",
            # Clustering - silhouette_score, calinski_harabasz는 maximize, davies_bouldin은 minimize
            "silhouette_score": "maximize", "davies_bouldin": "minimize", "calinski_harabasz": "maximize",
            # Causal - 기본적으로 maximize
            "ate": "maximize", "att": "maximize", "confidence_intervals": "maximize"
        }
        
        direction = metric_directions.get(optimization_metric, "maximize")
        
        study = optuna_integration.create_study(
            direction=direction,
            study_name=f"study_{self.settings.recipe.name}",
            pruner=self.pruner
        )
        
        start_time = datetime.now()

        def objective(trial):
            # optuna_integration을 통해 파라미터 제안
            params = optuna_integration.suggest_hyperparameters(
                trial, self.settings.recipe.model.hyperparameters.tunable or {}
            )
            result = training_callback(train_df=train_df, params=params, seed=trial.number)
            
            # 선택된 optimization_metric에 해당하는 점수 반환
            if optimization_metric in result:
                score = result[optimization_metric]
            else:
                # fallback to 'score' key for backward compatibility
                score = result.get('score', 0.0)
            
            # Periodic output every 10 trials
            if (trial.number + 1) % 10 == 0 or trial.number + 1 == n_trials:
                self.console.info(
                    f"[OptunaOptimizer] 진행 상황: {trial.number + 1}/{n_trials} 시행, 현재 점수: {score:.4f}"
                )
            
            return score
        
        study.optimize(
            objective, 
            n_trials=self.settings.recipe.model.hyperparameters.n_trials or 10,
            timeout=self.settings.recipe.model.hyperparameters.timeout,
            callbacks=[logging_callback]
        )
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        self.console.log_model_operation(
            f"[OptunaOptimizer] 하이퍼파라미터 최적화가 완료되었습니다",
            {
                "optimization_metric": optimization_metric,
                "best_value": f"{study.best_value:.4f}",
                "optimization_time": f"{optimization_time:.1f}s",
                "total_trials": len(study.trials),
                "pruned_trials": len([t for t in study.trials if t.state.name == 'PRUNED'])
            }
        )

        return {
            'enabled': True,
            'engine': 'optuna',
            'optimization_metric': optimization_metric,
            'optimization_direction': direction,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state.name == 'PRUNED']),
            'optimization_time': optimization_time,
            'search_space': self.settings.recipe.model.hyperparameters.tunable or {}
        }

# Self registration under TrainerRegistry (same layer)
TrainerRegistry.register("optuna", OptunaOptimizer)
