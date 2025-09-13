"""
Business Validator - 비즈니스 로직 검증

Recipe와 Config 모델에서 분리한 비즈니스 로직 검증을 담당합니다.
데이터 무결성, 설정 일관성, 비즈니스 규칙 준수 등을 검증합니다.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
import re
from src.utils.core.logger import logger

class ValidationError(Exception):
    """Validation 에러"""
    pass

class BusinessValidator:
    """
    비즈니스 로직 검증기

    원래 Pydantic 모델의 @field_validator와 @model_validator에 있던
    비즈니스 로직 검증을 독립적으로 수행합니다.
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
        logger.error(f"[business_validator] {message}")

    def add_warning(self, message: str):
        """경고 메시지 추가"""
        self.warnings.append(message)
        logger.warning(f"[business_validator] {message}")

    def validate_data_split_ratios(self, split_config: Dict[str, float]) -> bool:
        """데이터 분할 비율의 합이 1.0인지 검증"""
        train = split_config.get('train', 0)
        validation = split_config.get('validation', 0)
        test = split_config.get('test', 0)
        calibration = split_config.get('calibration', 0)

        total = train + validation + test + calibration
        if abs(total - 1.0) > 0.001:
            self.add_error(
                f"데이터 분할 비율의 합이 1.0이어야 합니다. 현재 합: {total:.3f} "
                f"(train: {train}, validation: {validation}, "
                f"test: {test}, calibration: {calibration})"
            )
            return False

        # 각 비율이 0 이상 1 이하인지 검증
        for name, ratio in [('train', train), ('validation', validation),
                           ('test', test), ('calibration', calibration)]:
            if ratio < 0 or ratio > 1:
                self.add_error(f"{name} ratio must be between 0 and 1, got {ratio}")
                return False

        return True

    def validate_hyperparameter_tuning(self, hp_config: Dict[str, Any]) -> bool:
        """하이퍼파라미터 튜닝 설정 검증"""
        tuning_enabled = hp_config.get('tuning_enabled', False)

        if tuning_enabled:
            # 튜닝 활성화시 필수 필드 검증
            required_fields = ['optimization_metric', 'direction', 'n_trials']
            for field in required_fields:
                if field not in hp_config or hp_config[field] is None:
                    self.add_error(f"Hyperparameter tuning is enabled but '{field}' is missing")
                    return False

            # direction 값 검증
            direction = hp_config.get('direction')
            if direction not in ['minimize', 'maximize']:
                self.add_error(f"Direction must be 'minimize' or 'maximize', got '{direction}'")
                return False

            # n_trials 범위 검증
            n_trials = hp_config.get('n_trials')
            if not isinstance(n_trials, int) or n_trials < 1 or n_trials > 1000:
                self.add_error(f"n_trials must be between 1 and 1000, got {n_trials}")
                return False

            # timeout 검증 (선택사항)
            timeout = hp_config.get('timeout')
            if timeout is not None:
                if not isinstance(timeout, int) or timeout < 10:
                    self.add_error(f"timeout must be at least 10 seconds, got {timeout}")
                    return False

            # tunable 파라미터 구조 검증
            tunable = hp_config.get('tunable')
            if tunable:
                if not self.validate_tunable_parameters(tunable):
                    return False

        else:
            # 튜닝 비활성화시 values 필드 존재 확인
            values = hp_config.get('values')
            if values is None:
                self.add_warning("Tuning is disabled but 'values' field is missing")

        return True

    def validate_tunable_parameters(self, tunable: Dict[str, Dict[str, Any]]) -> bool:
        """튜닝 가능한 파라미터 구조 검증"""
        for param_name, spec in tunable.items():
            if not isinstance(spec, dict):
                self.add_error(f"Tunable parameter '{param_name}' must be a dictionary")
                return False

            # type 필드 검증
            param_type = spec.get('type')
            if param_type not in ['int', 'float', 'categorical']:
                self.add_error(
                    f"Parameter '{param_name}' type must be 'int', 'float', or 'categorical', "
                    f"got '{param_type}'"
                )
                return False

            # range 필드 검증
            param_range = spec.get('range')
            if param_range is None:
                self.add_error(f"Parameter '{param_name}' is missing 'range' field")
                return False

            # 타입별 range 형식 검증
            if param_type in ['int', 'float']:
                if not isinstance(param_range, list) or len(param_range) != 2:
                    self.add_error(
                        f"Parameter '{param_name}' range must be a list of 2 elements for {param_type} type"
                    )
                    return False

                if param_range[0] >= param_range[1]:
                    self.add_error(
                        f"Parameter '{param_name}' range minimum ({param_range[0]}) must be "
                        f"less than maximum ({param_range[1]})"
                    )
                    return False

            elif param_type == 'categorical':
                if not isinstance(param_range, list) or len(param_range) == 0:
                    self.add_error(
                        f"Parameter '{param_name}' range must be a non-empty list for categorical type"
                    )
                    return False

        return True

    def validate_calibration_settings(self, calibration_config: Optional[Dict[str, Any]]) -> bool:
        """캘리브레이션 설정 검증"""
        if not calibration_config:
            return True

        enabled = calibration_config.get('enabled', False)
        method = calibration_config.get('method')

        if enabled and not method:
            self.add_error(
                "캘리브레이션이 활성화된 경우 method를 지정해야 합니다. "
                "사용 가능: 'beta', 'isotonic', 'temperature'"
            )
            return False

        # method 값 검증
        if method and method not in ['beta', 'isotonic', 'temperature']:
            self.add_error(
                f"Calibration method must be 'beta', 'isotonic', or 'temperature', "
                f"got '{method}'"
            )
            return False

        return True

    def validate_preprocessor_step(self, step: Dict[str, Any]) -> bool:
        """개별 전처리 스텝 비즈니스 로직 검증"""
        step_type = step.get('type')

        # SimpleImputer 검증
        if step_type == 'simple_imputer':
            strategy = step.get('strategy')
            if not strategy:
                self.add_error("simple_imputer requires 'strategy' field")
                return False

            if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
                self.add_error(
                    f"simple_imputer strategy must be 'mean', 'median', 'most_frequent', "
                    f"or 'constant', got '{strategy}'"
                )
                return False

        # KBinsDiscretizer 검증
        elif step_type == 'kbins_discretizer':
            strategy = step.get('strategy', 'quantile')  # 기본값
            if strategy not in ['uniform', 'quantile', 'kmeans']:
                self.add_error(
                    f"kbins_discretizer strategy must be 'uniform', 'quantile', "
                    f"or 'kmeans', got '{strategy}'"
                )
                return False

            n_bins = step.get('n_bins', 5)  # 기본값
            if not isinstance(n_bins, int) or n_bins < 2 or n_bins > 20:
                self.add_error(f"kbins_discretizer n_bins must be between 2 and 20, got {n_bins}")
                return False

        # PolynomialFeatures 검증
        elif step_type == 'polynomial_features':
            degree = step.get('degree', 2)  # 기본값
            if not isinstance(degree, int) or degree < 2 or degree > 5:
                self.add_error(f"polynomial_features degree must be between 2 and 5, got {degree}")
                return False

        # CatBoostEncoder 검증
        elif step_type == 'catboost_encoder':
            sigma = step.get('sigma')
            if sigma is not None:
                if not isinstance(sigma, (int, float)) or sigma < 0.0 or sigma > 1.0:
                    self.add_error(f"catboost_encoder sigma must be between 0.0 and 1.0, got {sigma}")
                    return False

        # create_missing_indicators는 simple_imputer에서만 유효
        create_missing = step.get('create_missing_indicators')
        if create_missing is not None and step_type != 'simple_imputer':
            self.add_error("create_missing_indicators can only be used with simple_imputer")
            return False

        # Global vs Targeted 전처리기 검증
        global_preprocessors = {'standard_scaler', 'min_max_scaler', 'robust_scaler'}
        targeted_preprocessors = {
            'one_hot_encoder', 'ordinal_encoder', 'catboost_encoder',
            'simple_imputer', 'polynomial_features', 'tree_based_feature_generator', 'kbins_discretizer'
        }

        columns = step.get('columns')

        if step_type in targeted_preprocessors:
            if not columns:
                self.add_error(f"'{step_type}' is a targeted preprocessor and requires 'columns' field")
                return False

        return True

    def validate_cross_validation_settings(self, validation_config: Dict[str, Any]) -> bool:
        """교차 검증 설정 검증"""
        method = validation_config.get('method', 'train_test_split')

        if method == 'cross_validation':
            n_folds = validation_config.get('n_folds', 5)  # 기본값
            if not isinstance(n_folds, int) or n_folds < 2 or n_folds > 10:
                self.add_error(f"Cross validation n_folds must be between 2 and 10, got {n_folds}")
                return False

        test_size = validation_config.get('test_size', 0.2)
        if not isinstance(test_size, (int, float)) or test_size < 0.1 or test_size > 0.5:
            self.add_error(f"test_size must be between 0.1 and 0.5, got {test_size}")
            return False

        return True

    def validate_evaluation_metrics(self, metrics: List[str]) -> bool:
        """평가 메트릭 검증"""
        if not metrics:
            self.add_error("At least one evaluation metric is required")
            return False

        # 메트릭 이름 정규화 (소문자로)
        normalized_metrics = [m.lower() for m in metrics]

        # 중복 메트릭 확인
        if len(set(normalized_metrics)) != len(normalized_metrics):
            duplicates = [m for m in normalized_metrics if normalized_metrics.count(m) > 1]
            self.add_warning(f"Duplicate metrics found: {set(duplicates)}")

        return True

    def validate_feature_view_configuration(self, feature_views: Dict[str, Any]) -> bool:
        """Feature View 설정 검증"""
        for view_name, view_config in feature_views.items():
            if not isinstance(view_config, dict):
                self.add_error(f"Feature view '{view_name}' must be a dictionary")
                return False

            # join_key 필수
            if 'join_key' not in view_config:
                self.add_error(f"Feature view '{view_name}' is missing 'join_key'")
                return False

            # features 필수
            features = view_config.get('features')
            if not features or not isinstance(features, list):
                self.add_error(f"Feature view '{view_name}' must have a non-empty 'features' list")
                return False

        return True

    def validate_recipe_business_logic(self, recipe_data: Dict[str, Any]) -> bool:
        """Recipe의 모든 비즈니스 로직 검증"""
        self.clear_messages()
        valid = True

        # 데이터 분할 검증
        data_config = recipe_data.get('data', {})
        split_config = data_config.get('split')
        if split_config:
            if not self.validate_data_split_ratios(split_config):
                valid = False

        # 하이퍼파라미터 튜닝 검증
        model_config = recipe_data.get('model', {})
        hp_config = model_config.get('hyperparameters')
        if hp_config:
            if not self.validate_hyperparameter_tuning(hp_config):
                valid = False

        # 캘리브레이션 검증
        calibration_config = model_config.get('calibration')
        if calibration_config:
            if not self.validate_calibration_settings(calibration_config):
                valid = False

        # 전처리 스텝 검증
        preprocessor_config = recipe_data.get('preprocessor', {})
        steps = preprocessor_config.get('steps', [])
        for i, step in enumerate(steps):
            if not self.validate_preprocessor_step(step):
                valid = False

        # 평가 설정 검증
        evaluation_config = recipe_data.get('evaluation', {})
        metrics = evaluation_config.get('metrics', [])
        if not self.validate_evaluation_metrics(metrics):
            valid = False

        validation_config = evaluation_config.get('validation', {})
        if validation_config:
            if not self.validate_cross_validation_settings(validation_config):
                valid = False

        # Feature Views 검증 (있는 경우)
        fetcher_config = data_config.get('fetcher', {})
        feature_views = fetcher_config.get('feature_views')
        if feature_views:
            if not self.validate_feature_view_configuration(feature_views):
                valid = False

        return valid

    def validate_config_business_logic(self, config_data: Dict[str, Any]) -> bool:
        """Config의 모든 비즈니스 로직 검증"""
        self.clear_messages()
        valid = True

        # Serving 포트 범위 검증
        serving_config = config_data.get('serving', {})
        if serving_config:
            port = serving_config.get('port', 8000)
            if not isinstance(port, int) or port < 1024 or port > 65535:
                self.add_error(f"Serving port must be between 1024 and 65535, got {port}")
                valid = False

            workers = serving_config.get('workers', 1)
            if not isinstance(workers, int) or workers < 1:
                self.add_error(f"Serving workers must be at least 1, got {workers}")
                valid = False

        # MLflow 설정 검증
        mlflow_config = config_data.get('mlflow', {})
        if mlflow_config:
            tracking_uri = mlflow_config.get('tracking_uri')
            if not tracking_uri:
                self.add_error("MLflow tracking_uri is required")
                valid = False

            experiment_name = mlflow_config.get('experiment_name')
            if not experiment_name:
                self.add_error("MLflow experiment_name is required")
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