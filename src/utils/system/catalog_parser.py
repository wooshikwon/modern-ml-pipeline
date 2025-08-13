# src/utils/system/catalog_parser.py
from pathlib import Path
import yaml
from typing import Dict, List, Any

def load_model_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """
    모델 카탈로그 YAML을 로드합니다.
    위치: src/cli/resources/model_catalog.yaml
    """
    try:
        current_dir = Path(__file__).parent
        catalog_path = current_dir.parent.parent / "cli" / "resources" / "model_catalog.yaml"
        if not catalog_path.exists():
            return {}
        return yaml.safe_load(catalog_path.read_text())
    except Exception:
        return {}