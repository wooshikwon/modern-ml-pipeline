"""
Inference Command Implementation
모듈화된 배치 추론 실행 명령

"""

import json
from typing import Optional, Dict, Any
import typer
from typing_extensions import Annotated

from src.settings import SettingsFactory
from src.pipelines.inference_pipeline import run_inference_pipeline
from src.utils.core.console import (
    cli_command_start, cli_command_success, cli_command_error,
    cli_step_complete, get_rich_console
)


def batch_inference_command(
    run_id: Annotated[
        str, 
        typer.Option("--run-id", help="추론에 사용할 MLflow Run ID")
    ],
    config_path: Annotated[
        str,
        typer.Option("--config-path", "-c", help="Config 파일 경로")
    ],
    data_path: Annotated[
        str,
        typer.Option("--data-path", "-d", help="추론 데이터 파일 경로")
    ],
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    배치 추론 실행 (Phase 5.3 리팩토링).
    
    MLflow에 저장된 모델을 사용하여 배치 추론을 수행합니다.
    --data-path로 추론 데이터를 직접 지정합니다.
    
    Args:
        run_id: MLflow Run ID (학습된 모델)
        config_path: Config YAML 파일 경로
        data_path: 추론 데이터 파일 경로
        context_params: 추가 파라미터 (JSON 형식)
    
    Examples:
        mmp batch-inference --run-id abc123 --config-path configs/prod.yaml --data-path data/inference.csv
        mmp batch-inference --run-id abc123 -c configs/dev.yaml -d queries/inference.sql --params '{"date": "2024-01-01"}'
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        cli_command_start("Batch Inference", "배치 추론 파이프라인 실행")

        # 1. 파라미터 파싱
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )

        # 2. Settings 생성 과정 시각화
        console = get_rich_console()
        with console.progress_tracker("setup", 3, "추론 환경 설정") as update:
            settings = SettingsFactory.for_inference(
                run_id=run_id,
                config_path=config_path,
                data_path=data_path,
                context_params=params
            )
            update(3)  # 설정 완료

        cli_step_complete("설정", f"Config: {config_path}, Data: {data_path}, Run ID: {run_id}")

        # 3. 배치 추론 실행
        result = run_inference_pipeline(
            settings=settings,
            run_id=run_id,
            data_path=data_path,
            context_params=params or {},
        )

        # 4. 성공 완료 메시지
        success_details = []
        if hasattr(result, 'processed_rows'):
            success_details.append(f"처리된 데이터: {result.processed_rows}행")
        if hasattr(result, 'output_path'):
            success_details.append(f"출력 경로: {result.output_path}")

        cli_command_success("Batch Inference", success_details)

    except FileNotFoundError as e:
        cli_command_error("Batch Inference", f"파일을 찾을 수 없습니다: {e}",
                         "파일 경로를 확인하거나 올바른 Run ID를 사용하세요")
        raise typer.Exit(code=1)
    except ValueError as e:
        cli_command_error("Batch Inference", f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        cli_command_error("Batch Inference", f"실행 중 오류 발생: {e}")
        raise typer.Exit(code=1)