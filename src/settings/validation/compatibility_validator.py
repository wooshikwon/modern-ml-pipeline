"""
Compatibility Validator - 컴포넌트 간 호환성 검증

Recipe와 Config 간 호환성, Task 타입과 다른 설정들 간의 호환성,
Feature Store 설정과 Fetcher 설정 간의 호환성 등을 검증합니다.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Tuple
from src.utils.core.logger import logger

class ValidationError(Exception):
    """Validation 에러"""
    pass

class CompatibilityValidator:
    """
    컴포넌트 간 호환성 검증기

    여러 컴포넌트나 설정이 서로 호환되는지 검증합니다.
    특히 task_choice에 따른 다른 설정들의 호환성을 중점적으로 확인합니다.
    """

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def clear_messages(self):
        """에러와 경고 메시지 초기화"""
        self.errors.clear()
        self.warnings.clear()

    def add_error(self, message: str):
        """에러 메시지 추가"""
        self.errors.append(message)
        logger.error(f"[compatibility_validator] {message}")

    def add_warning(self, message: str):
        """경고 메시지 추가"""
        self.warnings.append(message)
        logger.warning(f"[compatibility_validator] {message}")

    def validate_task_choice_compatibility(self, recipe_data: Dict[str, Any]) -> bool:
        """task_choice와 다른 설정들의 호환성 검증"""
        task_choice = recipe_data.get('task_choice')
        if not task_choice:
            self.add_error("task_choice is required")
            return False

        data_config = recipe_data.get('data', {})
        data_interface = data_config.get('data_interface', {})
        model_config = recipe_data.get('model', {})

        valid = True

        # Clustering task 검증
        if task_choice == "clustering":
            target_column = data_interface.get('target_column')
            if target_column is not None:
                self.add_error("Clustering task에서는 target_column이 None이어야 합니다")
                valid = False

            # Clustering에서는 캘리브레이션 불가
            calibration = model_config.get('calibration', {})
            if calibration.get('enabled'):
                self.add_error("Clustering task에서는 캘리브레이션을 사용할 수 없습니다")
                valid = False

        # Classification/Regression task 검증
        elif task_choice in ["classification", "regression"]:
            target_column = data_interface.get('target_column')
            if not target_column:
                self.add_error(f"{task_choice.capitalize()} task에는 target_column이 필수입니다")
                valid = False

            # Regression에서는 캘리브레이션 불가
            if task_choice == "regression":
                calibration = model_config.get('calibration', {})
                if calibration.get('enabled'):
                    self.add_error("Regression task에서는 캘리브레이션을 사용할 수 없습니다")
                    valid = False

        # Timeseries task 검증
        elif task_choice == "timeseries":
            timestamp_column = data_interface.get('timestamp_column')
            if not timestamp_column:
                self.add_error("Timeseries task에는 timestamp_column이 필수입니다")
                valid = False

            target_column = data_interface.get('target_column')
            if not target_column:
                self.add_error("Timeseries task에는 target_column이 필수입니다")
                valid = False

        # Causal task 검증
        elif task_choice == "causal":
            treatment_column = data_interface.get('treatment_column')
            if not treatment_column:
                self.add_error("Causal task에는 treatment_column이 필수입니다")
                valid = False

            target_column = data_interface.get('target_column')
            if not target_column:
                self.add_error("Causal task에는 target_column이 필수입니다")
                valid = False

        return valid

    def validate_feature_store_compatibility(self, recipe_data: Dict[str, Any], config_data: Dict[str, Any]) -> bool:
        """Feature Store 설정과 Recipe 설정의 호환성 검증"""
        # Recipe의 fetcher 설정
        data_config = recipe_data.get('data', {})
        fetcher_config = data_config.get('fetcher', {})
        fetcher_type = fetcher_config.get('type')

        # Config의 feature store 설정
        feature_store_config = config_data.get('feature_store', {})
        fs_provider = feature_store_config.get('provider', 'none')

        valid = True

        # feature_store 타입 fetcher를 사용하는데 feature store가 비활성화된 경우
        if fetcher_type == 'feature_store' and fs_provider == 'none':
            self.add_error(
                "Recipe에서 feature_store fetcher를 사용하려면 "
                "Config에서 feature_store.provider를 'feast' 등으로 설정해야 합니다"
            )
            valid = False

        # Feast 설정이 있는데 feature_store fetcher를 사용하지 않는 경우 (경고)
        if fs_provider == 'feast' and fetcher_type != 'feature_store':
            self.add_warning(
                "Config에서 Feast를 설정했지만 Recipe에서 feature_store fetcher를 사용하지 않습니다. "
                "Feature Store의 이점을 활용하지 못할 수 있습니다"
            )

        # Feature Store 사용시 필요한 필드들 검증
        if fetcher_type == 'feature_store':
            feature_views = fetcher_config.get('feature_views')
            if not feature_views:
                self.add_warning("feature_store fetcher를 사용하지만 feature_views가 비어있습니다")

            # timestamp_column 검증
            timestamp_column = fetcher_config.get('timestamp_column')
            if not timestamp_column:
                self.add_warning(
                    "feature_store fetcher에서 timestamp_column이 없으면 "
                    "point-in-time join을 수행할 수 없습니다"
                )

        return valid

    def validate_data_split_calibration_compatibility(self, recipe_data: Dict[str, Any]) -> bool:
        """데이터 분할과 캘리브레이션 설정의 호환성 검증"""
        data_config = recipe_data.get('data', {})
        split_config = data_config.get('split', {})
        model_config = recipe_data.get('model', {})
        calibration_config = model_config.get('calibration', {})

        calibration_enabled = calibration_config.get('enabled', False)
        calibration_split = split_config.get('calibration', 0.0)

        valid = True

        # 캘리브레이션이 활성화되었는데 분할에서 캘리브레이션 비율이 0인 경우
        if calibration_enabled and calibration_split == 0.0:
            self.add_error(
                "Calibration이 활성화되었지만 data.split.calibration 비율이 0입니다. "
                "캘리브레이션을 위한 데이터를 분할해야 합니다"
            )
            valid = False

        # 캘리브레이션이 비활성화되었는데 분할에 캘리브레이션 비율이 있는 경우
        if not calibration_enabled and calibration_split > 0.0:
            self.add_warning(
                f"Calibration이 비활성화되었지만 data.split.calibration이 {calibration_split}로 설정되어 있습니다. "
                "불필요한 데이터 분할이 발생할 수 있습니다"
            )

        return valid

    def validate_hyperparameter_tuning_compatibility(self, recipe_data: Dict[str, Any]) -> bool:
        """하이퍼파라미터 튜닝 설정의 내부 호환성 검증"""
        model_config = recipe_data.get('model', {})
        hp_config = model_config.get('hyperparameters', {})

        tuning_enabled = hp_config.get('tuning_enabled', False)
        fixed_params = hp_config.get('fixed', {})
        tunable_params = hp_config.get('tunable', {})
        values_params = hp_config.get('values', {})

        valid = True

        # 튜닝 활성화시 fixed와 tunable 파라미터 중복 확인
        if tuning_enabled and fixed_params and tunable_params:
            fixed_keys = set(fixed_params.keys())
            tunable_keys = set(tunable_params.keys())
            overlap = fixed_keys & tunable_keys

            if overlap:
                self.add_error(
                    f"Fixed 파라미터와 tunable 파라미터가 중복됩니다: {overlap}. "
                    "각 파라미터는 fixed 또는 tunable 중 하나로만 설정해야 합니다"
                )
                valid = False

        # 튜닝 비활성화시 values와 다른 파라미터 설정의 중복 확인
        if not tuning_enabled:
            if fixed_params:
                self.add_warning(
                    "Tuning이 비활성화되었는데 'fixed' 파라미터가 설정되어 있습니다. "
                    "'values'를 사용하세요"
                )

            if tunable_params:
                self.add_warning(
                    "Tuning이 비활성화되었는데 'tunable' 파라미터가 설정되어 있습니다. "
                    "'values'를 사용하세요"
                )

        return valid

    def validate_evaluation_metrics_task_compatibility(self, recipe_data: Dict[str, Any]) -> bool:
        """평가 메트릭과 task 타입의 호환성 검증"""
        task_choice = recipe_data.get('task_choice')
        evaluation_config = recipe_data.get('evaluation', {})
        metrics = evaluation_config.get('metrics', [])

        # 태스크별 일반적인 메트릭 매핑
        task_metrics_map = {
            'classification': [
                'accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
                'roc_auc', 'roc_auc_macro', 'roc_auc_weighted', 'log_loss', 'matthews_corrcoef'
            ],
            'regression': [
                'mse', 'mae', 'rmse', 'r2', 'mean_absolute_error', 'mean_squared_error',
                'root_mean_squared_error', 'r2_score'
            ],
            'clustering': [
                'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
                'adjusted_rand_score', 'normalized_mutual_info_score'
            ],
            'timeseries': [
                'mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'mase'
            ],
            'causal': [
                'ate', 'pehe', 'abs_ate', 'policy_value', 'policy_risk'
            ]
        }

        expected_metrics = task_metrics_map.get(task_choice, [])
        valid = True

        # 메트릭이 해당 태스크에 적합한지 검증
        for metric in metrics:
            metric_lower = metric.lower()

            # Classification 메트릭이 다른 태스크에 사용되는 경우
            if (task_choice != 'classification' and
                metric_lower in task_metrics_map.get('classification', [])):
                self.add_warning(
                    f"Metric '{metric}'는 주로 classification 태스크에서 사용됩니다. "
                    f"현재 태스크: {task_choice}"
                )

            # Regression 메트릭이 다른 태스크에 사용되는 경우
            if (task_choice not in ['regression', 'timeseries'] and
                metric_lower in task_metrics_map.get('regression', [])):
                self.add_warning(
                    f"Metric '{metric}'는 주로 regression/timeseries 태스크에서 사용됩니다. "
                    f"현재 태스크: {task_choice}"
                )

            # Clustering 메트릭이 다른 태스크에 사용되는 경우
            if (task_choice != 'clustering' and
                metric_lower in task_metrics_map.get('clustering', [])):
                self.add_warning(
                    f"Metric '{metric}'는 주로 clustering 태스크에서 사용됩니다. "
                    f"현재 태스크: {task_choice}"
                )

        return valid

    def validate_adapter_config_compatibility(self, config_data: Dict[str, Any]) -> bool:
        """어댑터 설정의 호환성 검증"""
        data_source = config_data.get('data_source', {})
        adapter_type = data_source.get('adapter_type')
        adapter_config = data_source.get('config', {})

        valid = True

        # SQL 어댑터 필수 설정 검증
        if adapter_type == 'sql':
            connection_uri = adapter_config.get('connection_uri')
            if not connection_uri:
                self.add_error("SQL adapter requires 'connection_uri' in config")
                valid = False

        # Storage 어댑터 필수 설정 검증
        elif adapter_type == 'storage':
            base_path = adapter_config.get('base_path')
            if not base_path:
                self.add_error("Storage adapter requires 'base_path' in config")
                valid = False

        return valid

    def validate_recipe_config_compatibility(self, recipe_data: Dict[str, Any], config_data: Dict[str, Any]) -> bool:
        """Recipe와 Config 간 전체 호환성 검증"""
        self.clear_messages()
        valid = True

        # Task choice 호환성
        if not self.validate_task_choice_compatibility(recipe_data):
            valid = False

        # Feature Store 호환성
        if not self.validate_feature_store_compatibility(recipe_data, config_data):
            valid = False

        # 데이터 분할 및 캘리브레이션 호환성
        if not self.validate_data_split_calibration_compatibility(recipe_data):
            valid = False

        # 하이퍼파라미터 튜닝 호환성
        if not self.validate_hyperparameter_tuning_compatibility(recipe_data):
            valid = False

        # 평가 메트릭 호환성
        if not self.validate_evaluation_metrics_task_compatibility(recipe_data):
            valid = False

        # 어댑터 설정 호환성
        if not self.validate_adapter_config_compatibility(config_data):
            valid = False

        return valid

    def get_validation_summary(self) -> Dict[str, Any]:
        """검증 결과 요약 반환"""
        return {
            'is_valid': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
        }

    def suggest_task_appropriate_metrics(self, task_choice: str) -> List[str]:
        """태스크에 적합한 메트릭 제안"""
        suggestions = {
            'classification': ['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
            'regression': ['mse', 'mae', 'r2', 'rmse'],
            'clustering': ['silhouette_score', 'calinski_harabasz_score'],
            'timeseries': ['mse', 'mae', 'r2', 'mape'],
            'causal': ['ate', 'pehe', 'abs_ate']
        }
        return suggestions.get(task_choice, [])