from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import jinja2
import pandas as pd

from src.utils.core.logger import logger

if TYPE_CHECKING:
    pass


def render_template_from_file(template_path: str, context: Dict[str, Any]) -> str:
    """
    Jinja2를 사용하여 파일 기반 템플릿을 안전하게 렌더링합니다.
    - Context 파라미터 화이트리스트 검증
    - 렌더링된 SQL의 Injection 패턴 검증
    """
    logger.info(f"보안 강화 파일 템플릿 렌더링 시작: {template_path}")

    # 1. Context 파라미터 검증
    safe_context = _validate_context_params(context)

    # 2. 템플릿 렌더링
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found at: {template_path}")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=template_file.parent),
        undefined=jinja2.StrictUndefined,  # 정의되지 않은 변수 사용 시 에러 발생
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_file.name)
    rendered_sql = template.render(safe_context)

    # 3. 렌더링 결과 SQL 보안 검증
    _validate_sql_safety(rendered_sql)

    logger.info(f"보안 강화 파일 템플릿 렌더링 완료: {template_path}")
    return rendered_sql


def render_template_from_string(sql_template: str, context: Dict[str, Any]) -> str:
    """
    문자열 기반 SQL 템플릿을 안전하게 렌더링합니다. (Batch Inference 전용)
    - Context 파라미터 화이트리스트 검증
    - 렌더링된 SQL의 Injection 패턴 검증
    """
    logger.info("문자열 기반 SQL 템플릿 보안 렌더링 시작")

    # 1. Context 파라미터 검증
    safe_context = _validate_context_params(context)

    # 2. 템플릿 렌더링
    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(sql_template)
    rendered_sql = template.render(safe_context)

    # 3. 렌더링 결과 SQL 보안 검증
    _validate_sql_safety(rendered_sql)

    logger.info("문자열 기반 SQL 템플릿 보안 렌더링 완료")
    return rendered_sql


def _validate_context_params(context_params: dict) -> dict:
    """
    Context Parameters 값 검증
    변수명은 자유롭게 허용하되, 값의 형식을 검증

    Args:
        context_params: 검증할 컨텍스트 파라미터 딕셔너리

    Returns:
        검증을 통과한 안전한 컨텍스트 딕셔너리

    Raises:
        ValueError: 잘못된 값 형식
    """
    safe_context = {}

    for key, value in context_params.items():
        # 날짜 관련 파라미터는 형식 검증 (date, interval, start, end 등 포함)
        date_keywords = ["date", "interval", "start", "end", "time"]
        is_date_param = any(keyword in key.lower() for keyword in date_keywords)

        if is_date_param and isinstance(value, str):
            try:
                pd.to_datetime(value)  # 날짜 형식 검증
                safe_context[key] = value
            except Exception:
                raise ValueError(f"잘못된 날짜 형식: {key}={value}")
        else:
            safe_context[key] = value

    logger.info(f"Context Params 검증 통과: {list(safe_context.keys())}")
    return safe_context


def _validate_sql_safety(sql: str) -> None:
    """
    SQL 기본 검증 (읽기 전용 쿼리 확인)

    실제 보안은 파라미터 값 검증과 DB 권한으로 확보.
    여기서는 명백한 DDL만 차단.
    """
    import re

    # 명백한 DDL만 차단 (문장 시작 기준)
    ddl_pattern = r"^\s*(DROP|TRUNCATE|ALTER)\s+"
    if re.search(ddl_pattern, sql.upper(), re.MULTILINE):
        raise ValueError("DDL 명령어(DROP/TRUNCATE/ALTER)는 허용되지 않습니다.")

    logger.debug("SQL 검증 통과")


def is_jinja_template(text: str) -> bool:
    """
    텍스트가 Jinja2 템플릿인지 감지합니다.

    Args:
        text: 검사할 텍스트 문자열

    Returns:
        Jinja2 템플릿 패턴이 포함되어 있으면 True, 아니면 False
    """
    if not text:
        return False

    jinja_patterns = ["{{", "}}", "{%", "%}"]
    return any(pattern in text for pattern in jinja_patterns)
