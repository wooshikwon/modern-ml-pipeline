"""
Train Command Implementation
모듈화된 학습 파이프라인 실행 명령
"""

import json
from typing import Optional, Dict, Any
import typer
from typing_extensions import Annotated

from src.settings import SettingsFactory
from src.pipelines.train_pipeline import run_train_pipeline
from src.utils.core.logger import setup_logging, logger


def train_command(
    recipe_path: Annotated[
        str, 
        typer.Option("--recipe-path", "-r", help="Recipe 파일 경로")
    ],
    config_path: Annotated[
        str,
        typer.Option("--config-path", "-c", help="Config 파일 경로")
    ],
    data_path: Annotated[
        str,
        typer.Option("--data-path", "-d", help="학습 데이터 파일 경로")
    ],
    context_params: Annotated[
        Optional[str], 
        typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
    record_reqs: Annotated[
        bool,
        typer.Option("--record-reqs", help="현재 환경의 패키지 요구사항을 MLflow 아티팩트에 기록합니다(기본 비활성화).")
    ] = False,
) -> None:
    """
    학습 파이프라인 실행.
    
    Recipe와 Config 파일을 직접 지정하고, --data-path로 학습 데이터를 직접 전달합니다.
    DataInterface 기반 컬럼 검증을 수행한 후 학습을 진행합니다.
    
    Args:
        recipe_path: Recipe YAML 파일 경로
        config_path: Config YAML 파일 경로
        data_path: 학습 데이터 파일 경로
        context_params: 추가 파라미터 (JSON 형식)
    
    Examples:
        mmp train --recipe-path recipes/model.yaml --config-path configs/dev.yaml --data-path data/train.csv
        mmp train -r recipes/model.yaml -c configs/prod.yaml -d data/train.parquet --params '{"date": "2024-01-01"}'
        
    Raises:
        typer.Exit: 파일을 찾을 수 없거나 실행 중 오류 발생 시
    """
    try:
        # 1. 파라미터 파싱
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )

        # 2. Settings 생성 (모든 복잡성이 Factory에 내장됨)
        settings = SettingsFactory.for_training(
            recipe_path=recipe_path,
            config_path=config_path,
            data_path=data_path,
            context_params=params
        )
        setup_logging(settings)

        # 3. 학습 정보 로깅
        logger.info(f"Recipe: {recipe_path}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        computed = settings.recipe.model.computed
        run_name = computed.get("run_name", "unknown") if computed else "unknown"
        logger.info(f"Run Name: {run_name}")

        # 4. 학습 파이프라인 실행
        if record_reqs:
            run_train_pipeline(settings=settings, context_params=params, record_requirements=True)
        else:
            run_train_pipeline(settings=settings, context_params=params)
        
        logger.info("✅ 학습이 성공적으로 완료되었습니다.")

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"환경 설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"학습 파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)