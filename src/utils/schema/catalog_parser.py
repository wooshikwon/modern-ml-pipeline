# src/utils/system/catalog_parser.py
from pathlib import Path
import yaml
from typing import Dict, List, Any

def load_model_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """
    src/models/catalog.yaml 파일을 로드하고 파싱하여,
    카테고리별 모델 목록을 딕셔너리로 반환합니다.
    """
    try:
        # __file__을 기준으로 catalog.yaml의 절대 경로를 찾습니다.
        # 이렇게 하면 어떤 위치에서 코드가 실행되더라도 파일을 찾을 수 있습니다.
        current_dir = Path(__file__).parent
        catalog_path = current_dir.parent.parent / "models" / "catalog.yaml"

        if not catalog_path.exists():
            return {}
        
        return yaml.safe_load(catalog_path.read_text())
    except Exception:
        # 오류 발생 시 빈 카탈로그를 반환하여 cli가 중단되지 않도록 합니다.
        return {} 