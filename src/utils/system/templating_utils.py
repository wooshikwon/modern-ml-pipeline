from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import jinja2

if TYPE_CHECKING:
    from src.settings import Settings

def render_sql_template(template_path: str, context: Dict[str, Any]) -> str:
    """
    Jinja2를 사용하여 SQL 템플릿을 렌더링합니다.

    Args:
        template_path: 템플릿 파일의 경로.
        context: 템플릿에 주입할 컨텍스트 변수 딕셔너리.

    Returns:
        렌더링된 SQL 쿼리 문자열.
    """
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found at: {template_path}")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=template_file.parent),
        undefined=jinja2.StrictUndefined # 정의되지 않은 변수 사용 시 에러 발생
    )
    
    template = env.get_template(template_file.name)
    return template.render(context) 