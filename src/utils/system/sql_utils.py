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

from src.utils.system.logger import logger


def render_sql(
    file_path: str, params: Union[Dict[str, Any], None] = None
) -> str:
    """
    SQL 파일을 로드하고 Jinja2 템플릿을 사용하여 파라미터를 주입합니다.
    """
    params = params or {}
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
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


def render_sql_template(template_path: str, params: Dict[str, Any]) -> str:
    """
    SQL 템플릿을 렌더링하는 함수 (render_sql의 별칭)
    """
    return render_sql(template_path, params)


def parse_select_columns(sql_query: str) -> List[str]:
    """
    SQL 쿼리의 SELECT 절 컬럼을 파싱하는 함수 (get_selected_columns의 별칭)
    """
    return get_selected_columns(sql_query)


def parse_feature_columns(sql_query: str) -> tuple[List[str], str]:
    """
    SQL에서 피처 컬럼과 JOIN 키를 추출합니다.
    
    Args:
        sql_query: 분석할 SQL 쿼리
    
    Returns:
        tuple: (feature_columns, join_key)
    """
    feature_columns = get_selected_columns(sql_query)
    
    # JOIN 키를 찾기 위해 간단한 파싱 (WHERE 절이나 JOIN 절에서 추출)
    join_key = ""
    lower_sql = sql_query.lower()
    
    # 간단한 패턴 매칭으로 JOIN 키 추출
    if "member_id" in lower_sql:
        join_key = "member_id"
    elif "user_id" in lower_sql:
        join_key = "user_id"
    elif "id" in lower_sql:
        join_key = "id"
    
    logger.info(f"피처 컬럼 {len(feature_columns)}개, JOIN 키: {join_key}")
    return feature_columns, join_key


def sql_to_kv_mapping(sql_query: str, member_ids: List[str]) -> Dict[str, Any]:
    """
    SQL을 Key-Value 조회로 변환하는 매핑 정보를 생성합니다.
    
    Args:
        sql_query: 변환할 SQL 쿼리
        member_ids: 조회할 멤버 ID 리스트
    
    Returns:
        dict: Key-Value 조회를 위한 매핑 정보
    """
    feature_columns, join_key = parse_feature_columns(sql_query)
    
    mapping = {
        "feature_columns": feature_columns,
        "join_key": join_key,
        "member_ids": member_ids,
        "key_pattern": "{column}:{member_id}"
    }
    
    logger.info(f"SQL-to-KV 매핑 생성: {len(feature_columns)}개 컬럼, {len(member_ids)}개 ID")
    return mapping 