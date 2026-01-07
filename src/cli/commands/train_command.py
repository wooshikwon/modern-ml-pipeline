"""
Train Command Implementation
모듈화된 학습 파이프라인 실행 명령
"""

import json
import logging
from typing import Any, Dict, Optional

import typer
from typing_extensions import Annotated

from src.cli.utils import CLIProgress
from src.cli.utils.system_checker import CheckStatus, SystemChecker
from src.pipelines.train_pipeline import run_train_pipeline
from src.settings import SettingsFactory, __version__
from src.utils.core.logger import get_current_log_file, log_config, log_sys, setup_logging

logger = logging.getLogger(__name__)


def train_command(
    recipe_path: Annotated[str, typer.Option("--recipe", "-r", help="Recipe 파일 경로")],
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config 파일 경로")],
    data_path: Annotated[
        Optional[str],
        typer.Option("--data", "-d", help="학습 데이터 파일 경로 (SQL fetcher 사용 시 생략 가능)"),
    ] = None,
    context_params: Annotated[
        Optional[str], typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
    record_reqs: Annotated[
        bool,
        typer.Option(
            "--record-reqs",
            help="현재 환경의 패키지 요구사항을 MLflow 아티팩트에 기록합니다(기본 비활성화).",
        ),
    ] = False,
) -> None:
    """
    학습 파이프라인 실행.

    Recipe와 Config 파일을 직접 지정하고, --data로 학습 데이터를 직접 전달합니다.
    DataInterface 기반 컬럼 검증을 수행한 후 학습을 진행합니다.

    Args:
        recipe_path: Recipe YAML 파일 경로
        config_path: Config YAML 파일 경로
        data_path: 학습 데이터 파일 경로
        context_params: 추가 파라미터 (JSON 형식)

    Examples:
        mmp train --recipe recipes/model.yaml --config configs/dev.yaml --data data/train.csv
        mmp train -r recipes/model.yaml -c configs/prod.yaml -d data/train.parquet --params '{"date": "2024-01-01"}'

    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    # verbose 모드 감지 (main_commands.py의 -v 옵션으로 설정된 로그 레벨 확인)
    verbose = logging.getLogger().level == logging.DEBUG
    progress = CLIProgress(total_steps=6, verbose=verbose)
    progress.header(__version__)

    try:
        # 1. 설정 로드
        progress.step_start("Loading config")
        log_config(f"Recipe: {recipe_path}")
        log_config(f"Config: {config_path}")
        if data_path:
            log_config(f"Data: {data_path}")
        params: Optional[Dict[str, Any]] = json.loads(context_params) if context_params else None
        settings = SettingsFactory.for_training(
            recipe_path=recipe_path,
            config_path=config_path,
            data_path=data_path,
            context_params=params,
        )
        progress.step_done()

        # 2. 패키지 의존성 체크
        progress.step_start("Checking dependencies")
        log_sys("패키지 검증 시작")
        config_dict = settings.config.model_dump() if hasattr(settings.config, "model_dump") else {}
        recipe_dict = settings.recipe.model_dump() if hasattr(settings.recipe, "model_dump") else {}
        env_name = getattr(getattr(settings.config, "environment", None), "name", "local")

        checker = SystemChecker(config_dict, env_name, config_path, recipe=recipe_dict)
        dep_result = checker.check_package_dependencies()

        if dep_result.status == CheckStatus.FAILED:
            progress.step_fail()
            missing = dep_result.details.get("missing", []) if dep_result.details else []
            logger.error(f"  {dep_result.message}")
            logger.error(f"  Missing: {', '.join(missing)}")
            logger.error(f"  Solution: {dep_result.solution}")
            raise typer.Exit(code=1)
        log_sys("모든 패키지 확인 완료")
        progress.step_done()

        # 3-6. 로깅 초기화 후 학습 파이프라인 실행
        setup_logging(settings)

        # 콜백 핸들러: 파이프라인 이벤트를 CLI 진행 상태로 변환
        event_to_step = {
            "loading_data": "Loading data",
            "training": "Training model",
            "training_optuna": "Training model (optuna)",
            "evaluating": "Evaluating",
            "saving": "Saving artifacts",
        }

        def handle_progress(event: str, stats: str = "") -> None:
            if event in event_to_step:
                progress.step_start(event_to_step[event])
            elif event.endswith("_done"):
                progress.step_done(stats)

        result = run_train_pipeline(
            settings=settings,
            context_params=params,
            record_requirements=record_reqs,
            on_progress=handle_progress,
        )

        # 결과 출력
        run_id = getattr(result, "run_id", None)
        mlflow_url = getattr(result, "mlflow_url", None)
        artifact_uri = getattr(result, "artifact_uri", None)
        log_artifact_uri = getattr(result, "log_artifact_uri", None)
        if run_id:
            progress.result(run_id, mlflow_url, artifact_uri, log_artifact_uri)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        progress.step_fail()
        logger.info("  Training cancelled by user")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        progress.step_fail()
        logger.error(f"  File not found: {e}")
        logger.error("  Run 'mmp get-config' or 'mmp get-recipe' to create files")
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
