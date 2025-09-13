"""
List Commands Implementation
사용 가능한 컴포넌트 목록 표시 명령

"""

import yaml
import typer
from pathlib import Path
from typing import Dict, Any

from src.components.adapter import AdapterRegistry
from src.components.evaluator import EvaluatorRegistry
from src.components.preprocessor.registry import PreprocessorStepRegistry
# Phase 1에서 schema.catalog_parser 모듈이 제거됨
from src.utils.core.logger import logger
from src.utils.core.console_manager import (
    cli_success, cli_error, cli_print
)


def list_adapters() -> None:
    """
    사용 가능한 모든 데이터 어댑터의 별명 목록을 출력합니다.
    
    데이터 어댑터는 다양한 데이터 소스(DB, 파일, 클라우드 등)에서
    데이터를 로드하는 컴포넌트입니다.
    """
    cli_success("Available Adapters:")
    available_items = sorted(AdapterRegistry.list_adapters().keys())
    for item in available_items:
        cli_print(f"  - [cyan]{item}[/cyan]")
    
    if not available_items:
        cli_print("  [dim](No adapters available)[/dim]")


def list_evaluators() -> None:
    """
    사용 가능한 모든 평가자의 별명 목록을 출력합니다.
    
    평가자는 모델의 성능을 측정하는 메트릭을 제공하는 컴포넌트입니다.
    Task별로 적절한 평가 메트릭이 제공됩니다.
    """
    cli_success("Available Evaluators:")
    available_items = sorted(EvaluatorRegistry.get_available_tasks())
    for item in available_items:
        cli_print(f"  - [cyan]{item}[/cyan]")
    
    if not available_items:
        cli_print("  [dim](No evaluators available)[/dim]")


def list_preprocessors() -> None:
    """
    사용 가능한 모든 전처리기 블록의 별명 목록을 출력합니다.
    
    전처리기는 데이터 변환 및 피처 엔지니어링을 수행하는 컴포넌트입니다.
    StandardScaler, OneHotEncoder 등이 포함됩니다.
    """
    cli_success("Available Preprocessor Steps:")
    available_items = sorted(PreprocessorStepRegistry.preprocessor_steps.keys())
    for item in available_items:
        cli_print(f"  - [cyan]{item}[/cyan]")
    
    if not available_items:
        cli_print("  [dim](No preprocessor steps available)[/dim]")


def _load_catalog_from_directory() -> Dict[str, Any]:
    """
    src/models/catalog/ 디렉토리에서 모델 카탈로그를 로드합니다.
    
    Returns:
        Dict[str, Any]: Task별 모델 정보를 담은 딕셔너리
    """
    catalog_dir = Path(__file__).parent.parent.parent / "models" / "catalog"
    if not catalog_dir.exists():
        return {}
    
    catalog = {}
    
    # 각 카테고리 디렉토리를 순회
    for category_dir in catalog_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            catalog[category_name] = []
            
            # 각 모델 YAML 파일을 순회
            for model_file in category_dir.glob("*.yaml"):
                try:
                    with open(model_file, "r", encoding="utf-8") as f:
                        model_data = yaml.safe_load(f)
                        if model_data:
                            catalog[category_name].append(model_data)
                except Exception as e:
                    logger.warning(f"모델 파일 로드 실패: {model_file}, 오류: {e}")
                    continue
    
    return catalog


def list_models() -> None:
    """
    src/models/catalog/ 디렉토리에 등록된 사용 가능한 모델 목록을 출력합니다.
    
    모델은 Task별로 그룹화되어 표시되며, 각 모델의 라이브러리 정보도 함께 표시됩니다.
    """
    cli_success("Available Models from Catalog:")
    
    # 새로운 디렉토리 구조에서 로드 시도
    catalog_dir = Path(__file__).parent.parent.parent / "models" / "catalog"
    if catalog_dir.exists():
        model_catalog = _load_catalog_from_directory()
    else:
        # Fallback: 빈 카탈로그 (Phase 1에서 catalog_parser 제거됨)
        model_catalog = {}
    
    if not model_catalog:
        cli_error("src/models/catalog/ 디렉토리나 catalog.yaml 파일을 찾을 수 없거나 내용이 비어있습니다.")
        raise typer.Exit(1)

    for category, models in model_catalog.items():
        cli_print(f"\n[bold cyan]--- {category} ---[/bold cyan]")
        for model_info in models:
            class_path = model_info.get('class_path', 'Unknown')
            library = model_info.get('library', 'Unknown')
            cli_print(f"  - [green]{class_path}[/green] [dim]({library})[/dim]")