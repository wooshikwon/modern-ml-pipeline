from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import jinja2
import pandas as pd
from src.utils.system.logger import logger

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


def render_sql_template_safe(template_path: str, context: Dict[str, Any]) -> str:
    """
    🆕 Phase 3: SQL Injection 완전 차단하는 보안 강화 템플릿 렌더링
    기존 render_sql_template + Context Params 화이트리스트 검증 + SQL 패턴 검증
    
    Args:
        template_path: 템플릿 파일의 경로
        context: 템플릿에 주입할 컨텍스트 변수 딕셔너리
        
    Returns:
        보안 검증을 통과한 렌더링된 SQL 쿼리 문자열
        
    Raises:
        ValueError: 허용되지 않는 context parameter 또는 SQL Injection 패턴 감지
    """
    logger.info(f"🔒 보안 강화 SQL 템플릿 렌더링 시작: {template_path}")
    
    # 1. Context Params 화이트리스트 검증
    safe_context = _validate_context_params(context)
    
    # 2. 기존 render_sql_template 호출 (검증된 Jinja2 + StrictUndefined)
    rendered_sql = render_sql_template(template_path, safe_context)
    
    # 3. 렌더링 결과 SQL Injection 패턴 검증
    _validate_sql_safety(rendered_sql)
    
    logger.info(f"✅ 보안 강화 SQL 템플릿 렌더링 완료: {template_path}")
    return rendered_sql


def render_sql_from_string_safe(sql_template: str, context: Dict[str, Any]) -> str:
    """
    🆕 Phase 3: 문자열 기반 SQL 템플릿 보안 렌더링 (Batch Inference 전용)
    MLflow Artifact에 저장된 loader_sql_snapshot의 안전한 동적 렌더링
    
    Args:
        sql_template: Jinja2 템플릿 SQL 문자열
        context: 템플릿에 주입할 컨텍스트 변수 딕셔너리
        
    Returns:
        보안 검증을 통과한 렌더링된 SQL 쿼리 문자열
        
    Raises:
        ValueError: 허용되지 않는 context parameter 또는 SQL Injection 패턴 감지
    """
    logger.info("🔒 문자열 기반 SQL 템플릿 보안 렌더링 시작")
    
    # 1. Context Params 화이트리스트 검증
    safe_context = _validate_context_params(context)
    
    # 2. 기존 Jinja2 환경 설정 재사용 (StrictUndefined)
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    template = env.from_string(sql_template)
    rendered_sql = template.render(safe_context)
    
    # 3. 렌더링 결과 SQL Injection 패턴 검증
    _validate_sql_safety(rendered_sql)
    
    logger.info("✅ 문자열 기반 SQL 템플릿 보안 렌더링 완료")
    return rendered_sql


def _validate_context_params(context_params: dict) -> dict:
    """
    Context Parameters 화이트리스트 검증
    허용된 파라미터만 통과하는 강력한 보안 검증
    
    Args:
        context_params: 검증할 컨텍스트 파라미터 딕셔너리
        
    Returns:
        검증을 통과한 안전한 컨텍스트 딕셔너리
        
    Raises:
        ValueError: 허용되지 않는 파라미터 또는 잘못된 형식
    """
    # 허용된 파라미터 화이트리스트 (Blueprint 원칙 1: 논리적 파라미터만)
    ALLOWED_KEYS = ['start_date', 'end_date', 'target_date', 'period', 'include_target']
    safe_context = {}
    
    for key, value in context_params.items():
        if key not in ALLOWED_KEYS:
            raise ValueError(
                f"🚨 보안 위반: 허용되지 않는 context parameter '{key}'\n"
                f"허용된 파라미터: {ALLOWED_KEYS}"
            )
        
        # 날짜 파라미터 엄격한 형식 검증
        if 'date' in key:
            try:
                pd.to_datetime(value)  # 날짜 형식 검증, 실패 시 ValueError 발생
                safe_context[key] = value
            except Exception:
                raise ValueError(f"🚨 잘못된 날짜 형식: {key}={value}")
        else:
            safe_context[key] = value
    
    logger.info(f"✅ Context Params 검증 통과: {list(safe_context.keys())}")
    return safe_context


def _validate_sql_safety(sql: str) -> None:
    """
    SQL Injection 위험 패턴 완전 차단
    렌더링된 SQL에서 위험한 패턴을 감지하여 차단
    
    Args:
        sql: 검증할 SQL 쿼리 문자열
        
    Raises:
        ValueError: SQL Injection 패턴 감지 시
    """
    # SQL Injection 위험 패턴 (대소문자 무관)
    DANGEROUS_PATTERNS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE',
        'CREATE', 'EXEC', 'EXECUTE', '--', '/*', '*/', ';'
    ]
    
    sql_upper = sql.upper()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.upper() in sql_upper:
            raise ValueError(
                f"🚨 SQL Injection 패턴 감지: '{pattern}' found in SQL\n"
                f"허용되지 않는 SQL 명령어가 포함되어 있습니다."
            )
    
    logger.info("✅ SQL 보안 검증 통과") 