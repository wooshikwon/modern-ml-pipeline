"""
E2E Tests for CLI Config/Recipe Generation

현재 단위 테스트로 잡지 못하는 문제들:
1. Jinja2 템플릿 렌더링 후 YAML 문법 오류
2. Pydantic 경고 메시지
3. 다양한 설정 조합에서의 템플릿 오류
4. 안내 메시지와 실제 명령어 옵션 불일치

이 테스트는 실제 파일을 생성하고 검증합니다.
"""

import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.cli.utils.config_builder import InteractiveConfigBuilder
from src.cli.utils.template_engine import TemplateEngine

# ============================================================
# 1. 템플릿 렌더링 + YAML 파싱 테스트 (가장 중요)
# ============================================================


class TestConfigTemplateYamlValidity:
    """다양한 설정 조합으로 템플릿 렌더링 후 YAML 파싱 가능 여부 확인"""

    @pytest.fixture
    def template_engine(self):
        templates_dir = Path(__file__).parents[2] / "src" / "cli" / "templates"
        return TemplateEngine(templates_dir)

    # 모든 설정 조합 정의
    CONFIG_COMBINATIONS = [
        # (env_name, use_mlflow, data_source, feature_store, artifact_storage, inference_output)
        ("local", True, "Local Files", "없음", "Local", "Local Files"),
        ("local", False, "Local Files", "없음", None, "Local Files"),
        ("dev", True, "PostgreSQL", "없음", "Local", "PostgreSQL"),
        ("dev", True, "PostgreSQL", "Feast", "S3", "PostgreSQL"),
        ("prod", True, "BigQuery", "없음", "GCS", "BigQuery"),
        ("prod", True, "S3", "없음", "S3", "S3"),
        ("prod", True, "GCS", "Feast", "GCS", "GCS"),
        # MLflow 없는 케이스들
        ("local", False, "PostgreSQL", "없음", None, "Local Files"),
        ("local", False, "BigQuery", "없음", None, "BigQuery"),
        # Inference output 없는 케이스
        ("dev", True, "Local Files", "없음", "Local", None),
    ]

    @pytest.mark.parametrize(
        "env_name,use_mlflow,data_source,feature_store,artifact_storage,inference_output",
        CONFIG_COMBINATIONS,
    )
    def test_config_template_produces_valid_yaml(
        self,
        template_engine,
        tmp_path,
        env_name,
        use_mlflow,
        data_source,
        feature_store,
        artifact_storage,
        inference_output,
    ):
        """각 설정 조합이 유효한 YAML을 생성하는지 확인"""
        # Given: 설정 컨텍스트
        context = {
            "env_name": env_name,
            "use_mlflow": use_mlflow,
            "data_source": data_source,
            "feature_store": feature_store,
            "use_feast": feature_store == "Feast",
            "artifact_storage": artifact_storage,
            "inference_output_enabled": inference_output is not None,
            "inference_output_source": inference_output,
            "enable_serving": False,
            "serving_workers": 1,
            "model_stage": "None",
            "timestamp": "2025-01-01 00:00:00",
        }

        # Feast 관련 추가 컨텍스트
        if feature_store == "Feast":
            context["feast_registry_location"] = "로컬"
            context["feast_online_store"] = "SQLite"
            context["feast_needs_offline_path"] = True

        # When: 템플릿 렌더링
        output_path = tmp_path / f"{env_name}.yaml"
        template_engine.write_rendered_file("configs/config.yaml.j2", output_path, context)

        # Then: YAML 파싱 가능해야 함
        content = output_path.read_text()
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(
                f"YAML 파싱 실패 - 조합: {env_name}/{data_source}/{feature_store}\n에러: {e}\n내용:\n{content}"
            )

        # 필수 키 존재 확인
        assert "environment" in parsed, f"environment 키 누락: {content}"
        assert "data_source" in parsed, f"data_source 키 누락: {content}"

        if use_mlflow:
            assert "mlflow" in parsed, f"mlflow 키 누락 (use_mlflow=True): {content}"


# ============================================================
# 2. 경고 메시지 캡처 테스트
# ============================================================


class TestNoWarningsOnImport:
    """CLI import 시 경고가 발생하지 않는지 확인"""

    def test_no_pydantic_warnings_on_cli_import(self):
        """CLI 모듈 import 시 Pydantic 경고가 없어야 함"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # CLI 모듈 reimport (캐시 무시)
            import importlib

            import src.cli.main_commands

            importlib.reload(src.cli.main_commands)

            # Pydantic 관련 경고 필터링
            pydantic_warnings = [
                warning
                for warning in w
                if "pydantic" in str(warning.filename).lower()
                or "protected namespace" in str(warning.message).lower()
            ]

            assert (
                len(pydantic_warnings) == 0
            ), f"Pydantic 경고 발생: {[str(w.message) for w in pydantic_warnings]}"

    def test_no_runtime_warnings_on_cli_execution(self):
        """CLI 실행 시 RuntimeWarning이 없어야 함"""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parents[2]),
        )

        assert "RuntimeWarning" not in result.stderr, f"RuntimeWarning 발생: {result.stderr}"
        assert "UserWarning" not in result.stderr, f"UserWarning 발생: {result.stderr}"


# ============================================================
# 3. 실제 파일 생성 통합 테스트
# ============================================================


class TestConfigFileGeneration:
    """실제 파일 생성 후 검증"""

    @pytest.fixture
    def builder(self):
        return InteractiveConfigBuilder()

    @patch("builtins.input")
    def test_generates_valid_yaml_file(self, mock_input, tmp_path):
        """실제 config 파일 생성 후 YAML 파싱 가능 여부 확인"""
        # Given: 사용자 입력 시뮬레이션
        mock_input.side_effect = [
            "test",  # env_name
            "y",  # use_mlflow
            "./mlruns",  # mlflow_tracking_uri
            "3",  # data_source (Local Files)
            "1",  # feature_store (없음)
            "1",  # artifact_storage (Local)
            "y",  # inference_output_enabled
            "3",  # inference_output_source (Local Files)
            "y",  # confirm
        ]

        builder = InteractiveConfigBuilder()

        # When: 대화형 플로우 실행 및 파일 생성
        with patch.object(Path, "cwd", return_value=tmp_path):
            selections = builder.run_interactive_flow()
            config_path = builder.generate_config_file(selections["env_name"], selections)

        # Then: 파일이 존재하고 YAML 파싱 가능해야 함
        assert config_path.exists(), f"Config 파일이 생성되지 않음: {config_path}"

        content = config_path.read_text()
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"생성된 YAML 파싱 실패:\n{e}\n내용:\n{content}")

        # 구조 검증
        assert parsed["environment"]["name"] == "test"
        assert parsed["data_source"]["adapter_type"] == "storage"


# ============================================================
# 4. subprocess 기반 E2E 테스트
# ============================================================


class TestCLISubprocess:
    """실제 CLI 프로세스 실행 테스트"""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parents[2]

    def test_cli_help_no_errors(self, project_root):
        """CLI --help 실행 시 에러 없음"""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        assert result.returncode == 0, f"CLI 실행 실패: {result.stderr}"
        assert "Commands" in result.stdout
        assert "get-config" in result.stdout
        assert "get-recipe" in result.stdout

    def test_list_commands_no_errors(self, project_root):
        """list 명령어들이 에러 없이 실행됨"""
        for cmd in ["models", "preprocessors", "adapters", "evaluators"]:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "list", cmd],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )

            assert result.returncode == 0, f"list {cmd} 실패: {result.stderr}"
            # 경고 없어야 함
            assert "Warning" not in result.stderr, f"경고 발생 (list {cmd}): {result.stderr}"

    def test_system_check_shows_correct_options(self, project_root):
        """system-check --help가 올바른 옵션을 표시"""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "system-check", "--help"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        assert result.returncode == 0
        assert "--config" in result.stdout or "-c" in result.stdout
        # --env-name은 없어야 함 (잘못된 옵션)
        assert "--env-name" not in result.stdout, "system-check에 --env-name 옵션이 있으면 안됨"

    def test_train_shows_correct_options(self, project_root):
        """train --help가 올바른 옵션을 표시"""
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "train", "--help"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        assert result.returncode == 0
        assert "--recipe" in result.stdout or "-r" in result.stdout
        assert "--config" in result.stdout or "-c" in result.stdout
        assert "--data" in result.stdout or "-d" in result.stdout


# ============================================================
# 5. 안내 메시지 검증 테스트
# ============================================================


class TestGuideMessagesConsistency:
    """안내 메시지가 실제 명령어 옵션과 일치하는지 확인"""

    def test_get_config_guide_uses_correct_options(self):
        """get-config 완료 후 안내 메시지가 올바른 옵션 사용"""
        # 안내 메시지 생성

        from src.cli.commands.get_config_command import _show_completion_message

        with patch("src.cli.commands.get_config_command.cli_success_panel") as mock_panel:
            _show_completion_message(
                "local", Path("configs/local.yaml"), Path(".env.local.template")
            )

            # 안내 내용 확인
            call_args = mock_panel.call_args[0][0]

            # 올바른 옵션 사용 확인
            assert (
                "system-check -c" in call_args or "system-check --config" in call_args
            ), f"system-check 안내가 잘못됨: {call_args}"
            assert (
                "--env-name" not in call_args
            ), f"잘못된 --env-name 옵션이 안내에 포함됨: {call_args}"

    def test_get_recipe_guide_uses_correct_options(self):
        """get-recipe 완료 후 안내 메시지가 올바른 옵션 사용"""
        from src.cli.commands.get_recipe_command import _show_success_message

        with patch("src.cli.commands.get_recipe_command.cli_success_panel") as mock_panel:
            selections = {
                "task_choice": "classification",
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "scikit-learn",
                },
            }
            _show_success_message(Path("recipes/test.yaml"), selections)

            call_args = mock_panel.call_args[0][0]

            # 올바른 옵션 확인
            assert (
                "-c configs/" in call_args or "--config" in call_args
            ), f"train 안내에 config 옵션이 없음: {call_args}"
            assert (
                "--env-name" not in call_args
            ), f"잘못된 --env-name 옵션이 안내에 포함됨: {call_args}"


# ============================================================
# 6. 설정 조합 매트릭스 테스트
# ============================================================


class TestConfigCombinationMatrix:
    """모든 설정 조합에서 에러가 발생하지 않는지 확인"""

    DATA_SOURCES = ["Local Files", "PostgreSQL", "BigQuery", "S3", "GCS"]
    FEATURE_STORES = ["없음", "Feast"]
    ARTIFACT_STORAGES = ["Local", "S3", "GCS"]
    MLFLOW_OPTIONS = [True, False]

    @pytest.mark.parametrize("data_source", DATA_SOURCES)
    @pytest.mark.parametrize("use_mlflow", MLFLOW_OPTIONS)
    def test_data_source_mlflow_combinations(self, data_source, use_mlflow, tmp_path):
        """데이터 소스 × MLflow 조합 테스트"""
        templates_dir = Path(__file__).parents[2] / "src" / "cli" / "templates"
        engine = TemplateEngine(templates_dir)

        context = {
            "env_name": "test",
            "use_mlflow": use_mlflow,
            "data_source": data_source,
            "feature_store": "없음",
            "use_feast": False,
            "artifact_storage": "Local" if use_mlflow else None,
            "inference_output_enabled": True,
            "inference_output_source": "Local Files",
            "enable_serving": False,
            "timestamp": "2025-01-01",
        }

        output_path = tmp_path / "test.yaml"
        engine.write_rendered_file("configs/config.yaml.j2", output_path, context)

        # YAML 파싱 가능해야 함
        content = output_path.read_text()
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            pytest.fail(f"YAML 오류 - {data_source}/{use_mlflow}: {e}\n{content}")
