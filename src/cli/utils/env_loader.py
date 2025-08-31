"""
Environment Loader Utility
Phase 2: 환경별 설정 로드 및 환경변수 치환

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
import yaml

from src.utils.system.logger import logger


def load_environment(env_name: str, base_path: Optional[Path] = None) -> None:
    """
    환경변수 파일 로드.
    
    Args:
        env_name: 환경 이름
        base_path: 프로젝트 루트 경로 (테스트용)
        
    Raises:
        FileNotFoundError: .env.{env_name} 파일이 없을 때
    """
    base_path = base_path or Path.cwd()
    env_file = base_path / f".env.{env_name}"
    
    if not env_file.exists():
        raise FileNotFoundError(
            f".env.{env_name} 파일을 찾을 수 없습니다.\n"
            f"다음 명령어로 생성하세요:\n"
            f"  1. mmp get-config --env-name {env_name}\n"
            f"  2. cp .env.{env_name}.template .env.{env_name}\n"
            f"  3. .env.{env_name} 파일 편집"
        )
    
    load_dotenv(env_file, override=True)
    os.environ['ENV_NAME'] = env_name
    logger.info(f"환경변수 파일 로드됨: {env_file}")


def get_config_path(env_name: str, base_path: Optional[Path] = None) -> Path:
    """
    환경별 config 파일 경로 반환.
    
    Args:
        env_name: 환경 이름
        base_path: 프로젝트 루트 경로 (테스트용)
        
    Returns:
        Config 파일 경로
        
    Raises:
        FileNotFoundError: config 파일이 없을 때
    """
    base_path = base_path or Path.cwd()
    
    # configs/ 또는 config/ 디렉토리 확인
    for dir_name in ['configs', 'config']:
        config_file = base_path / dir_name / f"{env_name}.yaml"
        if config_file.exists():
            return config_file
    
    raise FileNotFoundError(
        f"configs/{env_name}.yaml 파일을 찾을 수 없습니다.\n"
        f"'mmp get-config --env-name {env_name}'로 생성하세요."
    )


def resolve_env_variables(value: Any) -> Any:
    """
    재귀적으로 환경변수 치환.
    ${VAR_NAME:default} 패턴 지원.
    
    Args:
        value: 치환할 값 (문자열, 딕셔너리, 리스트 등)
        
    Returns:
        환경변수가 치환된 값
    """
    if isinstance(value, str):
        # ${VAR:default} 패턴 매칭
        pattern = r'\$\{([^}]+)\}'
        
        # 문자열이 완전히 환경변수인지 확인
        full_match = re.fullmatch(pattern, value)
        
        if full_match:
            # 완전히 환경변수인 경우 타입 변환 시도
            expr = full_match.group(1)
            if ':' in expr:
                var_name, default_value = expr.split(':', 1)
                result = os.getenv(var_name.strip(), default_value.strip())
            else:
                result = os.getenv(expr.strip(), value)
            
            # 타입 변환
            if isinstance(result, str):
                if result.lower() in ('true', 'false'):
                    return result.lower() == 'true'
                try:
                    if '.' not in result:
                        return int(result)
                    return float(result)
                except (ValueError, AttributeError):
                    return result
            return result
        else:
            # 부분적으로 환경변수가 포함된 경우 문자열로 치환
            def replacer(match):
                expr = match.group(1)
                if ':' in expr:
                    var_name, default_value = expr.split(':', 1)
                    return str(os.getenv(var_name.strip(), default_value.strip()))
                else:
                    return str(os.getenv(expr.strip(), match.group(0)))
            
            return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: resolve_env_variables(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [resolve_env_variables(item) for item in value]
    
    return value


def load_config_with_env(env_name: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    환경별 config 로드 및 환경변수 치환.
    
    Args:
        env_name: 환경 이름
        base_path: 프로젝트 루트 경로 (테스트용)
        
    Returns:
        환경변수가 치환된 config 딕셔너리
    """
    # 1. 환경변수 로드
    load_environment(env_name, base_path)
    
    # 2. Config 파일 로드
    config_path = get_config_path(env_name, base_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 3. 환경변수 치환
    config = resolve_env_variables(config)
    
    logger.info(f"Config 로드 완료: {config_path}")
    return config


def get_env_name_with_fallback(
    env_name: Optional[str] = None,
    allow_env_var: bool = True
) -> str:
    """
    env_name을 가져오되, 없으면 환경변수에서 가져옴.
    
    Args:
        env_name: 명시적으로 전달된 환경 이름
        allow_env_var: 환경변수 fallback 허용 여부
        
    Returns:
        환경 이름
        
    Raises:
        ValueError: env_name이 없고 환경변수도 없을 때
    """
    if env_name:
        return env_name
    
    if allow_env_var:
        env_from_var = os.getenv('ENV_NAME')
        if env_from_var:
            logger.info(f"ENV_NAME 환경변수에서 환경 이름 사용: {env_from_var}")
            return env_from_var
    
    raise ValueError(
        "--env-name 파라미터가 필요합니다.\n"
        "또는 ENV_NAME 환경변수를 설정하세요."
    )