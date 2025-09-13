"""
Tests for Recipe Examples - Fixtures System

Phase 1에서 구현된 recipe_examples.py의 예제 데이터가
유효한 Pydantic 모델로 로드되는지 검증하고 예제 데이터 간 일관성을 확인합니다.
"""

import pytest
from typing import Dict, Any

from tests.fixtures.recipe_examples import (
    RECIPE_CLASSIFICATION_EXAMPLE,
    RECIPE_REGRESSION_EXAMPLE,
    RECIPE_CLUSTERING_EXAMPLE,
    RECIPE_EXAMPLES,
    HYPERPARAMETER_EXAMPLES,
    PREPROCESSOR_EXAMPLES,
    FEATURE_STORE_EXAMPLES
)
from src.settings.recipe import Recipe


class TestRecipeExamplesValidity:
    """Recipe 예제 데이터가 유효한 Pydantic 모델로 로드되는지 테스트"""

    def test_classification_example_loads_as_recipe(self):
        """Classification 예제가 Recipe 모델로 정상 로드되는지 검증"""
        recipe = Recipe(**RECIPE_CLASSIFICATION_EXAMPLE)

        assert recipe.name == "classification_rf"
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert recipe.model.library == "sklearn"
        assert recipe.model.hyperparameters.tuning_enabled is True
        assert recipe.data.data_interface.target_column == "label"
        assert recipe.evaluation.metrics == ["accuracy", "f1", "roc_auc"]

    def test_regression_example_loads_as_recipe(self):
        """Regression 예제가 Recipe 모델로 정상 로드되는지 검증"""
        recipe = Recipe(**RECIPE_REGRESSION_EXAMPLE)

        assert recipe.name == "regression_xgb"
        assert recipe.task_choice == "regression"
        assert recipe.model.class_path == "xgboost.XGBRegressor"
        assert recipe.model.library == "xgboost"
        assert recipe.model.hyperparameters.tuning_enabled is False
        assert recipe.data.data_interface.target_column == "price"
        assert recipe.evaluation.metrics == ["mae", "mse", "r2"]

    def test_clustering_example_loads_as_recipe(self):
        """Clustering 예제가 Recipe 모델로 정상 로드되는지 검증"""
        recipe = Recipe(**RECIPE_CLUSTERING_EXAMPLE)

        assert recipe.name == "clustering_kmeans"
        assert recipe.task_choice == "clustering"
        assert recipe.model.class_path == "sklearn.cluster.KMeans"
        assert recipe.model.library == "sklearn"
        assert recipe.model.hyperparameters.tuning_enabled is True
        assert recipe.data.data_interface.target_column is None  # clustering has no target
        assert recipe.evaluation.metrics == ["silhouette_score", "calinski_harabasz_score"]

    def test_all_recipe_examples_load_successfully(self):
        """RECIPE_EXAMPLES 딕셔너리의 모든 예제가 정상 로드되는지 검증"""
        for task_type, example_data in RECIPE_EXAMPLES.items():
            recipe = Recipe(**example_data)
            assert recipe.task_choice == task_type
            assert recipe.name is not None
            assert recipe.model.class_path is not None


class TestRecipeExamplesConsistency:
    """Recipe 예제 데이터 간 일관성 검증"""

    def test_hyperparameter_examples_consistency(self):
        """하이퍼파라미터 예제의 내부 일관성 검증"""
        # Tuning enabled 예제
        tuning_enabled = HYPERPARAMETER_EXAMPLES["tuning_enabled"]
        assert tuning_enabled["tuning_enabled"] is True
        assert "optimization_metric" in tuning_enabled
        assert "direction" in tuning_enabled
        assert "n_trials" in tuning_enabled
        assert "fixed" in tuning_enabled
        assert "tunable" in tuning_enabled

        # Tuning disabled 예제
        tuning_disabled = HYPERPARAMETER_EXAMPLES["tuning_disabled"]
        assert tuning_disabled["tuning_enabled"] is False
        assert "values" in tuning_disabled

    def test_preprocessor_examples_structure(self):
        """전처리 예제의 구조가 올바른지 검증"""
        full_pipeline = PREPROCESSOR_EXAMPLES["full_pipeline"]
        assert isinstance(full_pipeline, list)
        assert len(full_pipeline) > 0

        for step in full_pipeline:
            assert "type" in step
            assert step["type"] is not None

        minimal_pipeline = PREPROCESSOR_EXAMPLES["minimal_pipeline"]
        assert isinstance(minimal_pipeline, list)
        assert len(minimal_pipeline) == 1
        assert minimal_pipeline[0]["type"] == "standard_scaler"

    def test_feature_store_examples_structure(self):
        """Feature Store 예제의 구조가 올바른지 검증"""
        feast_config = FEATURE_STORE_EXAMPLES["feast_configuration"]
        assert feast_config["type"] == "feature_store"
        assert "feature_views" in feast_config
        assert "timestamp_column" in feast_config

        pass_through = FEATURE_STORE_EXAMPLES["pass_through"]
        assert pass_through["type"] == "pass_through"

    def test_task_specific_data_interface_consistency(self):
        """Task 타입별 data_interface 설정의 일관성 검증"""
        # Classification: target_column 필수
        cls_recipe = Recipe(**RECIPE_CLASSIFICATION_EXAMPLE)
        assert cls_recipe.data.data_interface.target_column is not None

        # Regression: target_column 필수
        reg_recipe = Recipe(**RECIPE_REGRESSION_EXAMPLE)
        assert reg_recipe.data.data_interface.target_column is not None

        # Clustering: target_column이 None이어야 함
        cluster_recipe = Recipe(**RECIPE_CLUSTERING_EXAMPLE)
        assert cluster_recipe.data.data_interface.target_column is None

    def test_data_split_ratios_sum_to_one(self):
        """모든 예제의 데이터 분할 비율 합이 1.0인지 검증"""
        for task_type, example_data in RECIPE_EXAMPLES.items():
            recipe = Recipe(**example_data)
            split = recipe.data.split

            total = split.train + split.validation + split.test + split.calibration
            assert abs(total - 1.0) < 0.001, f"Task {task_type}: split ratios sum to {total}"

    def test_evaluation_metrics_not_empty(self):
        """모든 예제에 평가 메트릭이 정의되어 있는지 검증"""
        for task_type, example_data in RECIPE_EXAMPLES.items():
            recipe = Recipe(**example_data)
            assert recipe.evaluation.metrics is not None
            assert len(recipe.evaluation.metrics) > 0, f"Task {task_type} has no evaluation metrics"


class TestRecipeExamplesSpecificValidation:
    """특정 Recipe 예제에 대한 상세한 검증"""

    def test_classification_tuning_parameters_structure(self):
        """Classification 예제의 튜닝 파라미터 구조 검증"""
        recipe = Recipe(**RECIPE_CLASSIFICATION_EXAMPLE)
        hp = recipe.model.hyperparameters

        assert hp.tuning_enabled is True
        assert hp.optimization_metric is not None
        assert hp.direction is not None
        assert hp.n_trials is not None
        assert hp.fixed is not None
        assert hp.tunable is not None

        # Tunable 파라미터 구조 검증
        for param_name, param_spec in hp.tunable.items():
            assert "type" in param_spec
            assert "range" in param_spec
            assert param_spec["type"] in ["int", "float", "categorical"]

    def test_regression_fixed_hyperparameters(self):
        """Regression 예제의 고정 하이퍼파라미터 검증"""
        recipe = Recipe(**RECIPE_REGRESSION_EXAMPLE)
        hp = recipe.model.hyperparameters

        assert hp.tuning_enabled is False
        assert hp.values is not None
        assert len(hp.values) > 0

        # XGBoost 필수 파라미터 확인
        assert "n_estimators" in hp.values
        assert "max_depth" in hp.values
        assert "learning_rate" in hp.values
        assert "random_state" in hp.values

    def test_clustering_model_configuration(self):
        """Clustering 예제의 모델 설정 검증"""
        recipe = Recipe(**RECIPE_CLUSTERING_EXAMPLE)

        # KMeans specific configuration
        assert recipe.model.class_path == "sklearn.cluster.KMeans"
        assert recipe.model.hyperparameters.tuning_enabled is True

        # Clustering 최적화 메트릭 확인
        assert recipe.model.hyperparameters.optimization_metric == "silhouette_score"
        assert recipe.model.hyperparameters.direction == "maximize"

    def test_preprocessor_step_configurations(self):
        """전처리 스텝 설정의 유효성 검증"""
        # Classification 예제의 전처리 스텝
        recipe = Recipe(**RECIPE_CLASSIFICATION_EXAMPLE)
        if hasattr(recipe, 'preprocessor') and recipe.preprocessor:
            steps = recipe.preprocessor.steps

            for step in steps:
                assert "type" in step
                # type별 필수 필드 검증
                if step["type"] == "standard_scaler":
                    # 컬럼이 지정되어 있는지 확인
                    assert "columns" in step
                elif step["type"] == "one_hot_encoder":
                    assert "columns" in step

    def test_feature_store_integration_examples(self):
        """Feature Store 통합 예제 검증"""
        cls_recipe = Recipe(**RECIPE_CLASSIFICATION_EXAMPLE)

        # Feature Store 사용시 설정 확인
        if cls_recipe.data.fetcher.type == "feature_store":
            assert cls_recipe.data.fetcher.feature_views is not None
            # Feature views가 딕셔너리 형태인지 확인
            if cls_recipe.data.fetcher.feature_views:
                for view_name, view_config in cls_recipe.data.fetcher.feature_views.items():
                    assert isinstance(view_config, dict)


class TestRecipeExamplesEdgeCases:
    """Recipe 예제의 경계 사례 및 특수 케이스 테스트"""

    def test_empty_preprocessor_steps_valid(self):
        """전처리 스텝이 없는 경우도 유효한지 검증"""
        # Regression 예제는 전처리 스텝이 없음
        recipe = Recipe(**RECIPE_REGRESSION_EXAMPLE)
        # preprocessor 필드가 없거나 빈 steps도 유효해야 함
        if hasattr(recipe, 'preprocessor'):
            if recipe.preprocessor and hasattr(recipe.preprocessor, 'steps'):
                # steps가 빈 리스트여도 유효
                assert isinstance(recipe.preprocessor.steps, list)

    def test_calibration_disabled_by_default(self):
        """캘리브레이션이 기본적으로 비활성화되어 있는지 확인"""
        for task_type, example_data in RECIPE_EXAMPLES.items():
            recipe = Recipe(**example_data)

            # Classification이 아닌 경우 캘리브레이션 비활성화
            if task_type != "classification":
                if hasattr(recipe.model, 'calibration') and recipe.model.calibration:
                    assert recipe.model.calibration.enabled is False

    def test_metadata_optional_fields(self):
        """메타데이터의 선택적 필드들이 올바르게 처리되는지 확인"""
        for task_type, example_data in RECIPE_EXAMPLES.items():
            recipe = Recipe(**example_data)

            # 메타데이터가 있는 경우 검증
            if hasattr(recipe, 'metadata') and recipe.metadata:
                # author, created_at, description은 있을 수 있음
                pass

    def test_data_loader_different_sources(self):
        """다양한 데이터 소스 URI 형태 검증"""
        examples = [
            RECIPE_CLASSIFICATION_EXAMPLE,  # sql/train_data.sql
            RECIPE_REGRESSION_EXAMPLE,      # data/housing.csv
            RECIPE_CLUSTERING_EXAMPLE       # data/customers.parquet
        ]

        for example in examples:
            recipe = Recipe(**example)
            source_uri = recipe.data.loader.source_uri
            assert source_uri is not None
            assert len(source_uri) > 0
            # 다양한 확장자나 형태를 허용
            assert isinstance(source_uri, str)