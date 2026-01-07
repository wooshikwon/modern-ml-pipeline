"""
MLflow Artifact 저장/복원 시스템 테스트
- Recipe, Config, SQL Artifact 저장 테스트
- Artifact 복원 및 Settings 재구성 테스트
- Full Override 정책 테스트
"""

import mlflow
import pytest

from src.settings import Config, Recipe
from src.settings.mlflow_restore import (
    MLflowArtifactRestorer,
    MLflowArtifactSaver,
    restore_all_from_mlflow,
    restore_config_from_mlflow,
    restore_recipe_from_mlflow,
    save_training_artifacts_to_mlflow,
)


class TestMLflowArtifactSaver:
    """Artifact 저장 테스트"""

    def test_save_training_artifacts_with_recipe_and_config(
        self, settings_builder, isolated_mlflow_tracking
    ):
        """Recipe, Config 저장 테스트"""
        # Given: 유효한 Settings
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_task("classification")
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        # When: MLflow run 내에서 저장
        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=None
            )
            run_id = run.info.run_id

        # Then: Artifact 저장 확인
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "training_artifacts")
        artifact_names = [a.path for a in artifacts]

        assert any("recipe_snapshot.yaml" in p for p in artifact_names)
        assert any("config_snapshot.yaml" in p for p in artifact_names)

    def test_save_training_artifacts_with_sql_file(
        self, settings_builder, isolated_mlflow_tracking, isolated_temp_directory
    ):
        """SQL 파일 저장 테스트"""
        # Given: SQL 파일 생성
        sql_file = isolated_temp_directory / "test_query.sql"
        sql_content = "SELECT * FROM test_table WHERE date = '{{ date }}'"
        sql_file.write_text(sql_content)

        settings = settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment").build()

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        # When: SQL 포함하여 저장
        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=str(sql_file)
            )
            run_id = run.info.run_id

        # Then: SQL artifact 저장 확인
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "training_artifacts")
        artifact_names = [a.path for a in artifacts]

        assert any("source_query.sql" in p for p in artifact_names)

    def test_save_training_artifacts_without_sql(self, settings_builder, isolated_mlflow_tracking):
        """SQL 없이 저장 테스트"""
        settings = settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment").build()

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        # When: SQL 없이 저장
        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=None  # SQL 없음
            )
            run_id = run.info.run_id

        # Then: Recipe, Config만 저장됨
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "training_artifacts")
        artifact_names = [a.path for a in artifacts]

        assert any("recipe_snapshot.yaml" in p for p in artifact_names)
        assert any("config_snapshot.yaml" in p for p in artifact_names)
        # SQL은 없어야 함
        assert not any("source_query.sql" in p for p in artifact_names)

    def test_read_sql_content_with_valid_file(self, isolated_temp_directory):
        """유효한 SQL 파일 읽기 테스트"""
        # Given: SQL 파일
        sql_file = isolated_temp_directory / "query.sql"
        expected_content = "SELECT id, name FROM users"
        sql_file.write_text(expected_content)

        # When: 파일 읽기
        content = MLflowArtifactSaver._read_sql_content(str(sql_file))

        # Then: 내용 일치
        assert content == expected_content

    def test_read_sql_content_with_jinja_template(self, isolated_temp_directory):
        """Jinja2 템플릿 SQL 파일 읽기 테스트"""
        # Given: .sql.j2 파일
        sql_file = isolated_temp_directory / "query.sql.j2"
        expected_content = "SELECT * FROM {{ table_name }} WHERE date = '{{ date }}'"
        sql_file.write_text(expected_content)

        # When: 파일 읽기
        content = MLflowArtifactSaver._read_sql_content(str(sql_file))

        # Then: Jinja 템플릿 그대로 저장
        assert content == expected_content
        assert "{{ table_name }}" in content

    def test_read_sql_content_with_nonexistent_file(self):
        """존재하지 않는 SQL 파일 처리 테스트"""
        # When: 존재하지 않는 파일
        content = MLflowArtifactSaver._read_sql_content("/nonexistent/path/query.sql")

        # Then: None 반환
        assert content is None

    def test_read_sql_content_with_non_sql_file(self):
        """SQL이 아닌 파일 처리 테스트"""
        # When: CSV 파일 경로
        content = MLflowArtifactSaver._read_sql_content("data.csv")

        # Then: None 반환 (SQL 파일만 처리)
        assert content is None


class TestMLflowArtifactRestorer:
    """Artifact 복원 테스트"""

    def test_restore_recipe_success(self, settings_builder, isolated_mlflow_tracking):
        """Recipe 복원 테스트"""
        # Given: 저장된 Recipe
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

        # When: Recipe 복원
        restored_recipe = restore_recipe_from_mlflow(run_id)

        # Then: 복원된 Recipe 검증
        assert isinstance(restored_recipe, Recipe)
        assert restored_recipe.task_choice == "classification"
        assert "RandomForestClassifier" in restored_recipe.model.class_path

    def test_restore_config_success(self, settings_builder, isolated_mlflow_tracking):
        """Config 복원 테스트"""
        # Given: 저장된 Config
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_environment("production")
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(recipe=settings.recipe, config=settings.config)
            run_id = run.info.run_id

        # When: Config 복원
        restored_config = restore_config_from_mlflow(run_id)

        # Then: 복원된 Config 검증
        assert isinstance(restored_config, Config)
        assert restored_config.environment.name == "production"

    def test_restore_all_success(
        self, settings_builder, isolated_mlflow_tracking, isolated_temp_directory
    ):
        """Recipe, Config, SQL 모두 복원 테스트"""
        # Given: 저장된 Artifacts
        sql_file = isolated_temp_directory / "query.sql"
        sql_content = "SELECT * FROM test_table"
        sql_file.write_text(sql_content)

        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_task("regression")
            .build()
        )

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=str(sql_file)
            )
            run_id = run.info.run_id

        # When: 모두 복원
        restored_recipe, restored_config, restored_sql = restore_all_from_mlflow(run_id)

        # Then: 모두 복원됨
        assert isinstance(restored_recipe, Recipe)
        assert isinstance(restored_config, Config)
        assert restored_sql == sql_content

    def test_restore_recipe_not_found(self, isolated_mlflow_tracking):
        """Recipe 없을 때 에러 테스트"""
        mlflow.set_tracking_uri(isolated_mlflow_tracking)

        # When/Then: 존재하지 않는 run_id로 복원 시도
        with pytest.raises(ValueError, match="Recipe"):
            restore_recipe_from_mlflow("nonexistent_run_id_12345")

    def test_restore_sql_returns_none_when_not_saved(
        self, settings_builder, isolated_mlflow_tracking
    ):
        """SQL 없이 저장된 경우 None 반환 테스트"""
        # Given: SQL 없이 저장
        settings = settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment").build()

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(
                recipe=settings.recipe, config=settings.config, source_uri=None
            )
            run_id = run.info.run_id

        # When: SQL 복원 시도
        restorer = MLflowArtifactRestorer(run_id)
        sql = restorer.restore_sql()

        # Then: None 반환
        assert sql is None

    def test_env_variable_resolution(self, settings_builder, isolated_mlflow_tracking, monkeypatch):
        """환경변수 치환 테스트"""
        # Given: 환경변수 설정
        monkeypatch.setenv("TEST_DB_HOST", "production-db.example.com")

        settings = settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment").build()

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(recipe=settings.recipe, config=settings.config)
            run_id = run.info.run_id

        # When: 복원
        restorer = MLflowArtifactRestorer(run_id)

        # Test env variable resolution helper
        test_data = "${TEST_DB_HOST:default-host}"
        resolved = restorer._resolve_env_variables(test_data)

        # Then: 환경변수 치환됨
        assert resolved == "production-db.example.com"

    def test_env_variable_default_value(self):
        """환경변수 기본값 테스트"""
        restorer = MLflowArtifactRestorer("dummy_run_id")

        # Given: 존재하지 않는 환경변수 with 기본값
        test_data = "${NONEXISTENT_VAR:default_value}"

        # When: 치환
        resolved = restorer._resolve_env_variables(test_data)

        # Then: 기본값 사용
        assert resolved == "default_value"


class TestArtifactRoundTrip:
    """저장 -> 복원 라운드트립 테스트"""

    def test_recipe_roundtrip_preserves_all_fields(
        self, settings_builder, isolated_mlflow_tracking
    ):
        """Recipe 필드 보존 테스트"""
        # Given: 복잡한 Recipe
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_task("classification")
            .with_model(
                "sklearn.ensemble.GradientBoostingClassifier",
                hyperparameters={"n_estimators": 100, "max_depth": 5, "random_state": 42},
            )
            .with_entity_columns(["user_id", "product_id"])
            .with_target_column("is_purchased")
            .build()
        )

        original_recipe = settings.recipe

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(recipe=original_recipe, config=settings.config)
            run_id = run.info.run_id

        # When: 복원
        restored_recipe = restore_recipe_from_mlflow(run_id)

        # Then: 모든 필드 보존
        assert restored_recipe.task_choice == original_recipe.task_choice
        assert restored_recipe.model.class_path == original_recipe.model.class_path
        assert restored_recipe.data.data_interface.target_column == "is_purchased"
        assert "user_id" in restored_recipe.data.data_interface.entity_columns
        assert "product_id" in restored_recipe.data.data_interface.entity_columns

    def test_config_roundtrip_preserves_all_fields(
        self, settings_builder, isolated_mlflow_tracking
    ):
        """Config 필드 보존 테스트"""
        # Given: 복잡한 Config
        settings = (
            settings_builder.with_mlflow(isolated_mlflow_tracking, "test_experiment")
            .with_environment("staging")
            .with_data_source("storage")
            .build()
        )

        original_config = settings.config

        mlflow.set_tracking_uri(isolated_mlflow_tracking)
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            save_training_artifacts_to_mlflow(recipe=settings.recipe, config=original_config)
            run_id = run.info.run_id

        # When: 복원
        restored_config = restore_config_from_mlflow(run_id)

        # Then: 모든 필드 보존
        assert restored_config.environment.name == original_config.environment.name
        assert restored_config.mlflow.experiment_name == original_config.mlflow.experiment_name
