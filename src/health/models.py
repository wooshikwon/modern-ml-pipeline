"""
Health Check Models
Blueprint v17.0 - Data structures for system health validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 데이터 검증 (Pydantic)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class HealthStatus(Enum):
    """건강 상태 열거형"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy" 
    WARNING = "warning"
    UNKNOWN = "unknown"


class CheckCategory(Enum):
    """검사 카테고리 열거형"""
    ENVIRONMENT = "environment"
    MLFLOW = "mlflow"
    EXTERNAL_SERVICES = "external_services"
    TEMPLATES = "templates"
    SYSTEM = "system"


@dataclass
class CheckResult:
    """
    개별 건강 검사 결과를 나타내는 데이터 클래스.
    
    각 건강 검사의 성공/실패 여부와 상세 정보를 포함합니다.
    
    Attributes:
        is_healthy: 검사 통과 여부
        message: 주요 결과 메시지
        details: 상세 정보 목록 (선택사항)
        status: 건강 상태 (자동 계산)
        recommendations: 문제 해결을 위한 추천 사항들
    """
    is_healthy: bool
    message: str
    details: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    
    @property
    def status(self) -> HealthStatus:
        """건강 상태를 반환합니다."""
        return HealthStatus.HEALTHY if self.is_healthy else HealthStatus.UNHEALTHY


class HealthCheckConfig(BaseModel):
    """
    건강 검사 설정 모델.
    
    건강 검사 실행에 필요한 설정값들을 정의합니다.
    """
    
    # 연결 타임아웃 설정
    connection_timeout: int = Field(default=30, description="연결 타임아웃 (초)")
    
    # MLflow 설정
    mlflow_server_url: Optional[str] = Field(default=None, description="MLflow 서버 URL")
    mlflow_timeout: int = Field(default=10, description="MLflow 연결 타임아웃 (초)")
    
    # PostgreSQL 설정
    postgres_host: str = Field(default="localhost", description="PostgreSQL 호스트")
    postgres_port: int = Field(default=5432, description="PostgreSQL 포트")
    postgres_database: str = Field(default="modern_ml_pipeline", description="PostgreSQL 데이터베이스명")
    
    # Redis 설정
    redis_host: str = Field(default="localhost", description="Redis 호스트")
    redis_port: int = Field(default=6379, description="Redis 포트")
    
    # Feast 설정
    feast_repo_path: Optional[str] = Field(default=None, description="Feast repository 경로")
    
    # 출력 설정
    use_colors: bool = Field(default=True, description="컬러 출력 사용 여부")
    verbose: bool = Field(default=False, description="상세 출력 여부")
    
    class Config:
        """Pydantic 설정"""
        env_prefix = "MMP_HEALTH_"  # 환경 변수 접두사


@dataclass
class HealthCheckSummary:
    """
    전체 건강 검사 요약 정보.
    
    모든 카테고리의 건강 검사 결과를 통합한 요약을 제공합니다.
    """
    overall_healthy: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    categories: Dict[CheckCategory, CheckResult]
    execution_time_seconds: float
    timestamp: str
    
    @property
    def success_rate(self) -> float:
        """성공률을 퍼센티지로 반환합니다."""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100
    
    @property
    def has_warnings(self) -> bool:
        """경고가 있는지 확인합니다."""
        return self.warning_checks > 0


class RecommendationLevel(Enum):
    """추천 사항 우선순위 레벨"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Recommendation:
    """
    건강 검사 결과에 따른 추천 사항.
    
    문제 해결을 위한 구체적이고 실행 가능한 지침을 제공합니다.
    """
    level: RecommendationLevel
    title: str
    description: str
    action_command: Optional[str] = None
    documentation_url: Optional[str] = None
    category: Optional[CheckCategory] = None


class HealthCheckError(Exception):
    """
    건강 검사 중 발생하는 오류의 기본 클래스.
    
    건강 검사 실행 중 예상치 못한 오류가 발생했을 때 사용합니다.
    """
    
    def __init__(self, message: str, category: Optional[CheckCategory] = None, 
                 original_error: Optional[Exception] = None):
        """
        건강 검사 오류를 초기화합니다.
        
        Args:
            message: 오류 메시지
            category: 오류가 발생한 검사 카테고리
            original_error: 원본 예외 (있는 경우)
        """
        super().__init__(message)
        self.category = category
        self.original_error = original_error
        
    def __str__(self) -> str:
        """문자열 표현을 반환합니다."""
        category_str = f"[{self.category.value}] " if self.category else ""
        return f"{category_str}{super().__str__()}"


class ConnectionTestResult(BaseModel):
    """
    외부 서비스 연결 테스트 결과.
    
    데이터베이스, 캐시, 외부 API 등의 연결 상태를 나타냅니다.
    """
    
    service_name: str = Field(description="서비스명")
    is_connected: bool = Field(description="연결 성공 여부")
    response_time_ms: Optional[float] = Field(default=None, description="응답 시간 (밀리초)")
    error_message: Optional[str] = Field(default=None, description="오류 메시지")
    service_version: Optional[str] = Field(default=None, description="서비스 버전")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="추가 정보")
    
    @property
    def status_emoji(self) -> str:
        """상태를 나타내는 이모지를 반환합니다."""
        return "✅" if self.is_connected else "❌"
    
    @property
    def performance_rating(self) -> str:
        """응답 시간 기반 성능 등급을 반환합니다."""
        if not self.is_connected or self.response_time_ms is None:
            return "N/A"
        
        if self.response_time_ms < 100:
            return "Excellent"
        elif self.response_time_ms < 500:
            return "Good" 
        elif self.response_time_ms < 1000:
            return "Fair"
        else:
            return "Poor"