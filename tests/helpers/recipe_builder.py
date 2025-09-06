from typing import Dict, Any, Optional, List
from src.settings.recipe import Recipe

class RecipeBuilder:
    @staticmethod
    def build(
        name: str = 'test_recipe',
        model_class_path: str = 'sklearn.ensemble.RandomForestClassifier',
        task_type: str = 'classification',
        source_uri: str = './data/sample.csv',
        fetcher_type: str = 'pass_through',
        enable_tuning: bool = False,
        **overrides: Dict[str, Any]
    ) -> Recipe:
        if enable_tuning:
            hyperparameters = {
                'tuning_enabled': True,
                'fixed': {'random_state': 42},
                'tunable': {'n_estimators': {'type': 'int', 'range': [50, 200]}},
            }
        else:
            hyperparameters = {
                'tuning_enabled': False,
                'values': {'n_estimators': 100, 'random_state': 42},
            }
        metrics_map = {
            'classification': ['accuracy', 'precision', 'recall', 'f1'],
            'regression': ['mse', 'rmse', 'mae', 'r2'],
            'clustering': ['silhouette_score'],
            'causal': ['ate', 'uplift'],
        }
        metrics = metrics_map.get(task_type, ['accuracy'])
        target_column = 'cluster_label' if task_type == 'clustering' else 'target'
        recipe_dict: Dict[str, Any] = {
            'name': name,
            'model': {
                'class_path': model_class_path,
                'library': model_class_path.split('.')[0],
                'hyperparameters': hyperparameters,
            },
            'data': {
                'loader': {'source_uri': source_uri},
                'fetcher': {'type': fetcher_type},
                'data_interface': {
                    'task_type': task_type,
                    'target_column': target_column,
                    'entity_columns': ['user_id'],
                },
            },
            'evaluation': {
                'metrics': metrics,
                'validation': {'method': 'train_test_split', 'test_size': 0.2, 'random_state': 42},
            },
        }
        if task_type == 'causal':
            recipe_dict['data']['data_interface']['treatment_column'] = 'treatment'
        if fetcher_type == 'feature_store':
            recipe_dict['data']['fetcher']['timestamp_column'] = 'event_timestamp'
            recipe_dict['data']['fetcher']['feature_views'] = {}
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                current = recipe_dict
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                recipe_dict[key] = value
        return Recipe(**recipe_dict)
