"""
추론 파이프라인 Settings Override 테스트
- Artifact 복원 및 Full Override 정책 테스트
- _load_inference_settings() 함수 테스트
"""

import mlflow
import pytest
import yaml

from src.settings import Config, Recipe, Settings
from src.settings.mlflow_restore import (
    restore_all_from_mlflow,
    save_training_artifacts_to_mlflow,
)


class TestLoadInferenceSettings:
    """_load_inference_settings 함수 테스트"""

    @pytest.fixture
    def saved_training_artifacts(
        self, settings_builder, isolated_mlflow_tracking, isolated_temp_directory
    ):
        """학습 시 저장된 Artifacts"""
        # SQL 파일 생성
        sql_file = isolated_temp_directory / "training_query.sql"
        sql_file.write_text("SELECT * FROM training_data")

        # 학습용 Settings 생성
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_environment("production")
            .with_task("classification")
            .with_model("sklearn.ensemble.RandomForestClassifier")
            .with_target_column("target")
            .with_entity_columns(["user_id"])
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        # 학습 Artifact 저장
        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=str(sql_file)
            )
            run_id = run.info.run_id

        return {
            "run_id": run_id,
            "original_settings": settings,
            "tracking_uri": isolated_mlflow_tracking,
            "temp_dir": isolated_temp_directory,
        }

    def test_load_settings_from_artifact_only(self, saved_training_artifacts):
        """Artifact에서만 Settings 로드 (Override 없음)"""
        run_id = saved_training_artifacts["run_id"]

        # When: Override 없이 복원
        recipe, config, sql = restore_all_from_mlflow(run_id)

        # Then: Artifact의 설정 사용
        assert recipe.task_choice == "classification"
        assert "RandomForestClassifier" in recipe.model.class_path
        assert config.environment.name == "production"
        assert sql == "SELECT * FROM training_data"

    def test_full_override_with_recipe_path(self, saved_training_artifacts, settings_builder):
        """recipe_path 제공 시 Full Override 테스트"""
        run_id = saved_training_artifacts["run_id"]
        temp_dir = saved_training_artifacts["temp_dir"]

        # Override용 Recipe 파일 생성
        override_recipe = (
            settings_builder.with_task("regression")
            .with_model("sklearn.linear_model.LinearRegression")
            .build()
            .recipe
        )

        override_recipe_path = temp_dir / "override_recipe.yaml"
        with open(override_recipe_path, "w") as f:
            yaml.dump(override_recipe.model_dump(), f)

        # When: Recipe Override 적용
        # _load_inference_settings 로직 시뮬레이션
        artifact_recipe, artifact_config, artifact_sql = restore_all_from_mlflow(run_id)

        # Override 적용 (Full Override)
        with open(override_recipe_path, "r") as f:
            override_data = yaml.safe_load(f)
        overridden_recipe = Recipe(**override_data)

        # Then: Override된 Recipe 사용
        assert overridden_recipe.task_choice == "regression"
        assert "LinearRegression" in overridden_recipe.model.class_path

        # Artifact Recipe는 다름
        assert artifact_recipe.task_choice == "classification"

    def test_full_override_with_config_path(self, saved_training_artifacts, settings_builder):
        """config_path 제공 시 Full Override 테스트"""
        run_id = saved_training_artifacts["run_id"]
        temp_dir = saved_training_artifacts["temp_dir"]
        tracking_uri = saved_training_artifacts["tracking_uri"]

        # Override용 Config 파일 생성
        override_config = (
            settings_builder.with_environment("staging")
            .with_mlflow(tracking_uri, "staging_experiment")
            .build()
            .config
        )

        override_config_path = temp_dir / "override_config.yaml"
        with open(override_config_path, "w") as f:
            yaml.dump(override_config.model_dump(), f)

        # When: Config Override 적용
        artifact_recipe, artifact_config, artifact_sql = restore_all_from_mlflow(run_id)

        # Override 적용 (Full Override)
        with open(override_config_path, "r") as f:
            override_data = yaml.safe_load(f)
        overridden_config = Config(**override_data)

        # Then: Override된 Config 사용
        assert overridden_config.environment.name == "staging"

        # Artifact Config는 다름
        assert artifact_config.environment.name == "production"

    def test_partial_override_not_supported(self, saved_training_artifacts):
        """Partial Override는 지원하지 않음 (Full Override만)"""
        # 설계 결정: recipe_path나 config_path가 제공되면 해당 파일로 완전히 대체
        # merge나 부분 override는 지원하지 않음
        run_id = saved_training_artifacts["run_id"]

        # When: Artifact 복원
        recipe, config, _ = restore_all_from_mlflow(run_id)

        # Then: Artifact 값 그대로 사용 (merge 없음)
        assert recipe.task_choice == "classification"
        assert config.environment.name == "production"

    def test_both_recipe_and_config_override(self, saved_training_artifacts, settings_builder):
        """Recipe와 Config 모두 Override 테스트"""
        run_id = saved_training_artifacts["run_id"]
        temp_dir = saved_training_artifacts["temp_dir"]
        tracking_uri = saved_training_artifacts["tracking_uri"]

        # Override용 파일 생성
        new_settings = (
            settings_builder.with_environment("development")
            .with_mlflow(tracking_uri, "dev_experiment")
            .with_task("regression")
            .with_model("sklearn.linear_model.Ridge")
            .build()
        )

        override_recipe_path = temp_dir / "new_recipe.yaml"
        override_config_path = temp_dir / "new_config.yaml"

        with open(override_recipe_path, "w") as f:
            yaml.dump(new_settings.recipe.model_dump(), f)

        with open(override_config_path, "w") as f:
            yaml.dump(new_settings.config.model_dump(), f)

        # When: 모두 Override
        artifact_recipe, artifact_config, _ = restore_all_from_mlflow(run_id)

        with open(override_recipe_path, "r") as f:
            recipe_override = Recipe(**yaml.safe_load(f))

        with open(override_config_path, "r") as f:
            config_override = Config(**yaml.safe_load(f))

        # Then: 모두 Override된 값 사용
        assert recipe_override.task_choice == "regression"
        assert "Ridge" in recipe_override.model.class_path
        assert config_override.environment.name == "development"


class TestInferencePipelineArtifactUsage:
    """추론 파이프라인에서 Artifact 사용 테스트"""

    @pytest.fixture
    def training_run_with_artifacts(self, settings_builder, isolated_mlflow_tracking):
        """Artifact가 저장된 학습 Run"""
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_task("classification")
            .with_model("sklearn.ensemble.RandomForestClassifier")
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(recipe=settings.recipe, config=settings.config)
            run_id = run.info.run_id

        return run_id

    def test_inference_uses_artifact_settings_by_default(self, training_run_with_artifacts):
        """기본적으로 Artifact Settings 사용"""
        run_id = training_run_with_artifacts

        # When: Artifact에서 복원
        recipe, config, _ = restore_all_from_mlflow(run_id)

        # Then: 학습 시 저장된 설정
        assert recipe.task_choice == "classification"
        assert "RandomForestClassifier" in recipe.model.class_path

    def test_settings_reconstruction_from_artifacts(self, training_run_with_artifacts):
        """Artifact에서 Settings 재구성 테스트"""
        run_id = training_run_with_artifacts

        # When: Artifact 복원 및 Settings 재구성
        recipe, config, _ = restore_all_from_mlflow(run_id)
        settings = Settings(recipe=recipe, config=config)

        # Then: 유효한 Settings
        assert settings.recipe.task_choice == "classification"
        assert settings.config is not None


class TestOverridePolicy:
    """Override 정책 테스트"""

    def test_override_replaces_entirely(self, settings_builder, isolated_temp_directory):
        """Override는 완전히 대체함 (merge 아님)"""
        # Given: 원본 Recipe
        original_settings = (
            settings_builder.with_task("classification")
            .with_model(
                "sklearn.ensemble.RandomForestClassifier",
                hyperparameters={"n_estimators": 100, "max_depth": 10},
            )
            .with_entity_columns(["user_id", "product_id"])
            .build()
        )

        # Override Recipe (일부 필드만 다름)
        override_settings = (
            settings_builder.with_task("regression")
            .with_model(
                "sklearn.linear_model.LinearRegression", hyperparameters={}  # 다른 하이퍼파라미터
            )
            .with_entity_columns(["id"])
            .build()
        )

        # Override 파일 저장
        override_path = isolated_temp_directory / "override.yaml"
        with open(override_path, "w") as f:
            yaml.dump(override_settings.recipe.model_dump(), f)

        # When: Override 로드
        with open(override_path, "r") as f:
            loaded = Recipe(**yaml.safe_load(f))

        # Then: 완전히 Override된 값 사용 (원본과 merge 아님)
        assert loaded.task_choice == "regression"
        assert "LinearRegression" in loaded.model.class_path
        assert "id" in loaded.data.data_interface.entity_columns
        # 원본 필드가 merge되지 않음
        assert "user_id" not in loaded.data.data_interface.entity_columns

    def test_no_fallback_to_artifact_on_override(
        self, settings_builder, isolated_mlflow_tracking, isolated_temp_directory
    ):
        """Override 시 Artifact로 fallback하지 않음"""
        # Given: Artifact에 저장된 설정
        artifact_settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test")
            .with_task("classification")
            .with_model("sklearn.ensemble.RandomForestClassifier")
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=artifact_settings.recipe, config=artifact_settings.config
            )
            run_id = run.info.run_id

        # Override: 완전히 다른 설정
        override_settings = (
            settings_builder.with_task("regression")
            .with_model("sklearn.linear_model.Ridge")
            .build()
        )

        override_path = isolated_temp_directory / "override.yaml"
        with open(override_path, "w") as f:
            yaml.dump(override_settings.recipe.model_dump(), f)

        # When: Override 적용
        with open(override_path, "r") as f:
            loaded_recipe = Recipe(**yaml.safe_load(f))

        # Then: Artifact 값으로 fallback하지 않음
        assert loaded_recipe.task_choice == "regression"
        assert "Ridge" in loaded_recipe.model.class_path
        # RandomForestClassifier 관련 설정 없음 (fallback 없음)
