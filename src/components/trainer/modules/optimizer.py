# src/components/_trainer/_optimizer.py
from datetime import datetime
from typing import Dict, Any, Callable
import pandas as pd
from src.settings import Settings
from src.utils.system.logger import logger
from src.utils.integrations.optuna_integration import logging_callback

class OptunaOptimizer:
    def __init__(self, settings: Settings, factory_provider: Callable[[], Any]):
        self.settings = settings
        self.factory_provider = factory_provider
        self.pruner = self._create_pruner()

    def _create_pruner(self):
        """기본 Pruner 생성 (필요시 settings로 확장 가능)"""
        try:
            import optuna
            return optuna.pruners.MedianPruner()
        except Exception:
            return None

    def optimize(self, train_df: pd.DataFrame, training_callback: Callable) -> Dict[str, Any]:
        """Optuna를 사용하여 하이퍼파라미터 최적화를 수행합니다."""
        logger.info("🚀 하이퍼파라미터 자동 최적화 모드 시작")
        
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
                return result[optimization_metric]
            else:
                # fallback to 'score' key for backward compatibility
                return result.get('score', 0.0)
        
        study.optimize(
            objective, 
            n_trials=self.settings.recipe.model.hyperparameters.n_trials or 10,
            timeout=self.settings.recipe.model.hyperparameters.timeout,
            callbacks=[logging_callback]
        )
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 하이퍼파라미터 최적화 완료! 최고 {optimization_metric}: {study.best_value:.4f} ({optimization_time:.1f}초)")

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
