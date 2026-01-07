"""
DataHandler Catalog 유틸리티 함수.
Model catalog 로딩 및 DataHandler 타입 결정 로직을 담당한다.
"""

from pathlib import Path

from src.utils.core.logger import logger


def load_model_catalog(model_class_path: str) -> dict:
    """
    모델 클래스 경로에서 catalog 정보 로드.

    Args:
        model_class_path: 모델 클래스 경로 (예: sklearn.ensemble.RandomForestClassifier)

    Returns:
        catalog 딕셔너리 (로드 실패시 빈 딕셔너리)
    """
    if not model_class_path:
        return {}

    try:
        import yaml

        parts = model_class_path.split(".")
        if len(parts) >= 2:
            class_name = parts[-1]
            catalog_root = Path(__file__).parent.parent.parent / "models" / "catalog"

            for task_dir in catalog_root.iterdir():
                if task_dir.is_dir():
                    catalog_file = task_dir / f"{class_name}.yaml"
                    if catalog_file.exists():
                        with open(catalog_file, "r", encoding="utf-8") as f:
                            return yaml.safe_load(f) or {}

            logger.debug(f"[DATA:Catalog] 파일 없음: {class_name}")

    except Exception as e:
        logger.warning(f"[DATA:Catalog] 로드 실패: {model_class_path}, Error: {e}")

    return {}


def get_data_handler_type_from_catalog(model_class_path: str) -> str:
    """
    모델 catalog에서 data_handler 타입 추출.

    Args:
        model_class_path: 모델 클래스 경로

    Returns:
        사용할 data_handler 타입 (기본값: "tabular")
    """
    if not model_class_path:
        return "tabular"

    catalog = load_model_catalog(model_class_path)
    if catalog and "data_handler" in catalog:
        handler = catalog["data_handler"]
        logger.debug(f"[DATA:Catalog] data_handler: {handler}")
        return handler

    return "tabular"


def validate_task_handler_compatibility(task_choice: str, handler_type: str) -> None:
    """
    Task와 Handler 호환성 검증.
    호환성 문제 발생 시 경고 로그만 출력하고 예외는 발생시키지 않는다.

    Args:
        task_choice: Recipe의 task_choice
        handler_type: 선택된 handler 타입
    """
    if task_choice == "timeseries" and handler_type == "tabular":
        logger.warning("[DATA:Catalog] Timeseries task에 tabular handler 사용")
    elif handler_type == "sequence" and task_choice != "timeseries":
        logger.warning(
            f"[DATA:Catalog] Sequence handler는 timeseries task 전용 (현재: {task_choice})"
        )
