"""
List Commands Implementation
사용 가능한 컴포넌트 목록 표시 명령

"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import typer
import yaml

from src.components.adapter import AdapterRegistry
from src.components.evaluator import EvaluatorRegistry
from src.components.preprocessor.registry import PreprocessorStepRegistry

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


def _print_header(title: str) -> None:
    """헤더 출력"""
    sys.stdout.write(f"\nmmp v{VERSION}\n\n")
    sys.stdout.write(f"{title}\n")
    sys.stdout.flush()


def _print_section(name: str, items: list, show_library: bool = False, wide: bool = False) -> None:
    """섹션 출력"""
    sys.stdout.write(f"\n  {name}:\n")
    if items:
        width = 45 if wide else 30
        for item in items:
            if show_library and isinstance(item, tuple):
                sys.stdout.write(f"    - {item[0]:<{width}} ({item[1]})\n")
            else:
                sys.stdout.write(f"    - {item}\n")
    else:
        sys.stdout.write("    (none)\n")
    sys.stdout.flush()


def _print_summary(total: int, label: str) -> None:
    """요약 출력"""
    sys.stdout.write(f"\nTotal: {total} {label}\n")
    sys.stdout.flush()


def list_adapters() -> None:
    """
    사용 가능한 모든 데이터 어댑터의 별명 목록을 출력합니다.

    데이터 어댑터는 다양한 데이터 소스(DB, 파일, 클라우드 등)에서
    데이터를 로드하는 컴포넌트입니다.
    """
    _print_header("Adapters")

    available_items = sorted(AdapterRegistry.list_keys())
    for item in available_items:
        sys.stdout.write(f"  - {item}\n")

    if not available_items:
        sys.stdout.write("  (no adapters available)\n")

    _print_summary(len(available_items), "adapters")


def list_evaluators() -> None:
    """
    사용 가능한 모든 평가자의 별명 목록을 출력합니다.

    평가자는 모델의 성능을 측정하는 메트릭을 제공하는 컴포넌트입니다.
    Task별로 적절한 평가 메트릭이 제공됩니다.
    """
    _print_header("Evaluators")

    available_items = sorted(EvaluatorRegistry.list_keys())
    for item in available_items:
        sys.stdout.write(f"  - {item}\n")

    if not available_items:
        sys.stdout.write("  (no evaluators available)\n")

    _print_summary(len(available_items), "evaluators")


def list_metrics() -> None:
    """
    Task별 사용 가능한 평가 메트릭 목록을 출력합니다.

    각 Task(classification, regression 등)에서 사용할 수 있는
    optimization_metric 값들을 확인할 수 있습니다.
    """
    _print_header("Metrics by Task")

    available_tasks = sorted(EvaluatorRegistry.list_keys())

    if not available_tasks:
        sys.stdout.write("  (no evaluators registered)\n")
        return

    total_metrics = 0
    for task in available_tasks:
        metrics = EvaluatorRegistry.get_available_metrics_for_task(task)
        _print_section(task, metrics)
        total_metrics += len(metrics) if metrics else 0

    _print_summary(total_metrics, "metrics")


def list_preprocessors() -> None:
    """
    사용 가능한 모든 전처리기 블록의 별명 목록을 출력합니다.

    전처리기는 데이터 변환 및 피처 엔지니어링을 수행하는 컴포넌트입니다.
    StandardScaler, OneHotEncoder 등이 포함됩니다.
    """
    _print_header("Preprocessor Steps")

    available_items = sorted(PreprocessorStepRegistry.list_keys())
    for item in available_items:
        sys.stdout.write(f"  - {item}\n")

    if not available_items:
        sys.stdout.write("  (no preprocessor steps available)\n")

    _print_summary(len(available_items), "preprocessors")


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

    for category_dir in catalog_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            catalog[category_name] = []

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
    _print_header("Models by Task")

    catalog_dir = Path(__file__).parent.parent.parent / "models" / "catalog"
    if catalog_dir.exists():
        model_catalog = _load_catalog_from_directory()
    else:
        model_catalog = {}

    if not model_catalog:
        sys.stdout.write("  [ERROR] models/catalog/ directory not found or empty\n")
        raise typer.Exit(1)

    total_models = 0
    for category in sorted(model_catalog.keys()):
        models = model_catalog[category]
        model_items = []
        for model_info in models:
            class_path = model_info.get("class_path", "Unknown")
            library = model_info.get("library", "Unknown")
            model_items.append((class_path, library))

        _print_section(category, model_items, show_library=True, wide=True)
        total_models += len(models)

    sys.stdout.write("\n  Tip: class_path 값을 recipe의 model.class_path에 그대로 사용하세요.\n")
    _print_summary(total_models, "models")
