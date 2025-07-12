"""
SQL 유틸리티 모듈
- Jinja2 템플릿을 사용하여 SQL 파일을 렌더링합니다.
- sqlparse를 사용하여 SQL 쿼리를 파싱합니다.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional, List

import jinja2
import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import DML

from src.utils.logger import logger


def render_sql(
    file_path: str, params: Union[Dict[str, Any], None] = None
) -> str:
    """
    SQL 파일을 로드하고 Jinja2 템플릿을 사용하여 파라미터를 주입합니다.
    """
    params = params or {}
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(project_root)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        template = env.get_template(file_path)
        rendered_sql = template.render(params)
        logger.info(f"SQL 템플릿 렌더링 성공: {file_path}")
        return rendered_sql
    except jinja2.TemplateError as e:
        logger.error(f"SQL 템플릿 렌더링 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"SQL 템플릿 처리 중 예상치 못한 오류: {e}")
        raise


def get_selected_columns(sql_query: str) -> List[str]:
    """
    주어진 SQL 쿼리 문자열을 파싱하여 SELECT 절에 있는 컬럼명들을 추출합니다.
    'SELECT a, b, c FROM ...' -> ['a', 'b', 'c']
    'SELECT t1.a, t2.b as my_b, c FROM ...' -> ['a', 'my_b', 'c']
    """
    columns: List[str] = []
    parsed = sqlparse.parse(sql_query)[0]

    select_seen = False
    for token in parsed.tokens:
        if token.ttype is DML and token.normalized == "SELECT":
            select_seen = True
            continue
        if not select_seen:
            continue

        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                alias = identifier.get_alias()
                columns.append(alias if alias else identifier.get_real_name())
        elif isinstance(token, Identifier):
            alias = token.get_alias()
            columns.append(alias if alias else token.get_real_name())
        
        if columns:
            break
            
    logger.info(f"SQL에서 {len(columns)}개 컬럼 추출: {columns}")
    return columns