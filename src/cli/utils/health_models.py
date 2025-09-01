"""
System Check Models
CLI system-check 명령어 전용 데이터 구조

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 실제 사용 코드만 포함
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckResult:
    """
    시스템 검사 결과를 나타내는 데이터 클래스.
    
    CLI system-check 명령어에서 각 서비스(MLflow, PostgreSQL, Redis, Feature Store) 
    연결 검사 결과를 담는 핵심 데이터 구조입니다.
    
    Attributes:
        is_healthy: 검사 통과 여부
        message: 주요 결과 메시지
        details: 상세 정보 목록 (선택사항)
        recommendations: 문제 해결을 위한 추천 사항들 (선택사항)
        severity: 문제의 심각도 (critical, important, warning) (선택사항)
        
    Usage:
        success_result = CheckResult(
            is_healthy=True,
            message="MLflow Connection: 연결 성공"
        )
        
        failure_result = CheckResult(
            is_healthy=False,
            message="PostgreSQL Connection: 연결 실패",
            recommendations=["데이터베이스 서버 상태 확인"]
        )
    """
    is_healthy: bool
    message: str
    details: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    severity: Optional[str] = None  # critical, important, warning