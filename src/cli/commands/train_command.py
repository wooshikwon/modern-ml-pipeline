"""
Train Command Implementation
모듈화된 학습 파이프라인 실행 명령

"""

import json
from typing import Optional, Dict, Any
import typer
from typing_extensions import Annotated
from pathlib import Path

from src.settings import load_settings
from src.pipelines.train_pipeline import run_train_pipeline
from src.utils.system.logger import setup_logging, logger
from src.cli.utils.config_loader import load_environment


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
        # 1. Settings 생성 (recipe + config 직접 로드)
        params: Optional[Dict[str, Any]] = (
            json.loads(context_params) if context_params else None
        )
        settings = load_settings(recipe_path, config_path)
        setup_logging(settings)

        # 2. CLI data_path 처리
        if data_path.endswith('.sql.j2') or data_path.endswith('.sql') and params:
            # Jinja 템플릿 렌더링
            from src.utils.system.templating_utils import render_template_from_string
            from pathlib import Path
            
            template_path = Path(data_path)
            if template_path.exists():
                template_content = template_path.read_text()
                if params:
                    try:
                        rendered_sql = render_template_from_string(template_content, params)
                        logger.info(f"✅ Jinja 템플릿 렌더링 성공: {data_path}")
                        settings.recipe.data.loader.source_uri = rendered_sql
                    except ValueError as e:
                        logger.error(f"🚨 Jinja 렌더링 실패: {e}")
                        raise ValueError(f"템플릿 렌더링 실패: {e}")
                else:
                    # 파라미터 없이 .sql.j2 파일 → 에러
                    raise ValueError(f"Jinja 템플릿 파일({data_path})에는 --params가 필요합니다")
            else:
                raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {data_path}")
        else:
            # 일반 파일 경로 (CSV, Parquet, 정적 SQL 등)
            settings.recipe.data.loader.source_uri = data_path
        
        # 3. 데이터 소스 호환성 검증 (source_uri 주입 후)
        settings.validate_data_source_compatibility()
        
        # 4. 학습 정보 로깅
        logger.info(f"Recipe: {recipe_path}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        computed = settings.recipe.model.computed
        run_name = computed.get("run_name", "unknown") if computed else "unknown"
        logger.info(f"Run Name: {run_name}")
        
        # 4. 학습 파이프라인 실행
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