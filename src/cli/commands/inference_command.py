"""
Inference Command Implementation
모듈화된 배치 추론 실행 명령
"""

import json
import logging
from typing import Any, Dict, Optional

import typer
from typing_extensions import Annotated

from src.cli.utils import CLIProgress
from src.cli.utils.system_checker import CheckStatus, SystemChecker
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.settings import __version__
from src.settings.factory import SettingsFactory
from src.settings.mlflow_restore import restore_all_from_mlflow
from src.utils.core.logger import get_current_log_file, log_config, log_sys

logger = logging.getLogger(__name__)


def batch_inference_command(
    run_id: Annotated[str, typer.Option("--run-id", help="추론에 사용할 MLflow Run ID")],
    recipe_path: Annotated[
        Optional[str],
        typer.Option("--recipe", "-r", help="Recipe 파일 경로 (제공 시 artifact 대신 사용)"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config", "-c", help="Config 파일 경로 (제공 시 artifact 대신 사용)"),
    ] = None,
    data_path: Annotated[
        Optional[str],
        typer.Option(
            "--data", "-d", help="추론 데이터 파일 경로 (제공 시 artifact의 SQL 대신 사용)"
        ),
    ] = None,
    context_params: Annotated[
        Optional[str], typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    배치 추론 실행.

    MLflow에 저장된 모델과 설정을 사용하여 배치 추론을 수행합니다.
    기본적으로 학습 시 저장된 artifact의 Recipe, Config, SQL을 사용합니다.
    --recipe, --config, --data 옵션으로 완전 덮어쓰기(override) 가능합니다.

    Args:
        run_id: MLflow Run ID (학습된 모델)
        recipe_path: Override할 Recipe 파일 경로 (선택)
        config_path: Override할 Config 파일 경로 (선택)
        data_path: Override할 데이터 파일 경로 (선택)
        context_params: SQL 렌더링용 파라미터 (JSON 형식)

    Examples:
        # Artifact 설정 그대로 사용
        mmp inference --run-id abc123

        # Config만 override
        mmp inference --run-id abc123 --config configs/prod.yaml

        # Recipe와 Config 모두 override
        mmp inference --run-id abc123 --recipe recipes/new.yaml --config configs/prod.yaml

        # 데이터 소스 override
        mmp inference --run-id abc123 --data queries/new_data.sql --params '{"date": "2024-01-01"}'
    """
    # verbose 모드 감지 (main_commands.py의 -v 옵션으로 설정된 로그 레벨 확인)
    verbose = logging.getLogger().level == logging.DEBUG
    progress = CLIProgress(total_steps=4, verbose=verbose)
    progress.header(__version__)

    try:
        # 1. 설정 로드
        progress.step_start("Loading settings")
        log_config(f"Run ID: {run_id}")
        if recipe_path:
            log_config(f"Recipe override: {recipe_path}")
        if config_path:
            log_config(f"Config override: {config_path}")
        if data_path:
            log_config(f"Data override: {data_path}")
        params: Optional[Dict[str, Any]] = json.loads(context_params) if context_params else None

        factory_instance = SettingsFactory()
        artifact_recipe, artifact_config, _ = restore_all_from_mlflow(run_id)

        if recipe_path:
            recipe = factory_instance._load_recipe(recipe_path)
        else:
            recipe = artifact_recipe

        if config_path:
            config = factory_instance._load_config(config_path)
        else:
            config = artifact_config
        progress.step_done(f"run_id: {run_id[:8]}...")

        # 2. 패키지 의존성 체크
        progress.step_start("Checking package deps")
        log_sys("패키지 검증 시작")
        config_dict = config.model_dump() if hasattr(config, "model_dump") else {}
        recipe_dict = recipe.model_dump() if hasattr(recipe, "model_dump") else {}
        env_name = getattr(getattr(config, "environment", None), "name", "local")

        checker = SystemChecker(
            config_dict, env_name, config_path or "artifact", recipe=recipe_dict
        )
        dep_result = checker.check_package_dependencies()

        if dep_result.status == CheckStatus.FAILED:
            progress.step_fail()
            logger.error(f"  {dep_result.message}")
            logger.error(f"  Solution: {dep_result.solution}")
            raise typer.Exit(code=1)
        log_sys("모든 패키지 확인 완료")
        progress.step_done("verified")

        # 3. 배치 추론 실행
        progress.step_start("Executing inference")
        result = run_inference_pipeline(
            run_id=run_id,
            recipe_path=recipe_path,
            config_path=config_path,
            data_path=data_path,
            context_params=params or {},
        )
        progress.step_done(f"{result.prediction_count:,} rows")

        # 4. 결과 마무리
        progress.step_start("Finalizing results")
        progress.step_done()

        # 결과 출력
        progress.result(result.run_id)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        progress.step_fail()
        logger.info("  Inference cancelled by user")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        progress.step_fail()
        logger.error(f"  File not found: {e}")
        logger.error("  Check the file path or use a valid Run ID")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
    except ValueError as e:
        progress.step_fail()
        logger.error(f"  Configuration error: {e}")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        progress.step_fail()
        logger.error(f"  {e}")
        log_file = get_current_log_file()
        if log_file:
            logger.error(f"  See: {log_file}")
        raise typer.Exit(code=1)
