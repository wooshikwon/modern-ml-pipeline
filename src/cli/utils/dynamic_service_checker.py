"""
Dynamic Service Checker for Environment-specific System Checks
Phase 2: 환경별 시스템 체크 유틸리티

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from src.cli.utils.system_check_models import CheckResult
from src.utils.system.logger import logger


class DynamicServiceChecker:
    """
    환경별 설정에 따라 동적으로 시스템을 체크하는 클래스.
    
    단일 환경의 config를 받아서 실제 설정된 서비스만 체크합니다:
    - MLflow tracking_uri 기반 연결 테스트
    - PostgreSQL connection_uri 기반 연결 테스트  
    - Redis online_store 기반 연결 테스트
    - Feature Store feast_config 기반 설정 검증
    """
    
    def check_single_environment(
        self, 
        env_name: str, 
        config: Dict[str, Any], 
        actionable: bool = False
    ) -> Dict[str, Any]:
        """
        단일 환경의 시스템 체크 실행.
        
        Args:
            env_name: 환경 이름
            config: 환경별 config 딕셔너리
            actionable: 실행 가능한 해결책 제시 여부
            
        Returns:
            체크 결과 요약 딕셔너리
        """
        results = []
        
        # 1. MLflow 연결 체크
        mlflow_result = self._check_mlflow_connection(env_name, config)
        if mlflow_result:
            results.append(mlflow_result)
        
        # 2. PostgreSQL 연결 체크
        postgres_result = self._check_postgres_connection(env_name, config)
        if postgres_result:
            results.append(postgres_result)
        
        # 3. Redis 연결 체크
        redis_result = self._check_redis_connection(env_name, config)
        if redis_result:
            results.append(redis_result)
        
        # 4. Feature Store 설정 검증
        fs_result = self._check_feature_store_config(env_name, config)
        if fs_result:
            results.append(fs_result)
        
        # 요약 생성
        passed_count = sum(1 for r in results if r.is_healthy)
        failed_count = len(results) - passed_count
        
        summary = {
            'results': results,
            'overall_healthy': all(r.is_healthy for r in results),
            'total_checks': len(results),
            'passed_checks': passed_count,
            'failed_checks': failed_count,
            'environment': env_name
        }
        
        return summary
    
    def _check_mlflow_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        MLflow 서버 연결 테스트.
        
        Args:
            env: 환경 이름
            config: Config 딕셔너리
            
        Returns:
            CheckResult 또는 None (MLflow 미설정시)
        """
        mlflow_config = config.get('mlflow')
        if not mlflow_config or 'tracking_uri' not in mlflow_config:
            return None
            
        tracking_uri = mlflow_config['tracking_uri']
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # 로컬 파일 경로인 경우 체크 스킵
            if tracking_uri.startswith('./') or tracking_uri.startswith('/'):
                return CheckResult(
                    is_healthy=True,
                    message=f"MLflow ({env}): 로컬 스토리지 사용 - {tracking_uri}",
                    severity="info"
                )
            
            # 원격 서버 연결 테스트
            original_uri = mlflow.get_tracking_uri()
            try:
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient(tracking_uri)
                experiments = client.search_experiments(max_results=1)
                
                return CheckResult(
                    is_healthy=True,
                    message=f"MLflow ({env}): 연결 성공 - {tracking_uri}",
                    details=[f"실험 수: {len(experiments) if experiments else 0}"],
                    severity="info"
                )
            finally:
                mlflow.set_tracking_uri(original_uri)
                
        except Exception as e:
            suggestion = self._generate_mlflow_suggestion(tracking_uri, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"MLflow ({env}): 연결 실패 - {str(e)[:50]}",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_postgres_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        PostgreSQL 데이터베이스 연결 테스트.
        
        Args:
            env: 환경 이름
            config: Config 딕셔너리
            
        Returns:
            CheckResult 또는 None (PostgreSQL 미설정시)
        """
        adapters = config.get('data_adapters', {}).get('adapters', {})
        sql_adapter = adapters.get('sql', {})
        sql_config = sql_adapter.get('config', {})
        connection_uri = sql_config.get('connection_uri')
        
        if not connection_uri:
            return None
            
        try:
            import psycopg2
            
            # URI 파싱
            parsed = urlparse(connection_uri)
            
            if not parsed.hostname:
                return CheckResult(
                    is_healthy=False,
                    message=f"PostgreSQL ({env}): 잘못된 connection_uri 형식",
                    recommendations=["connection_uri 형식을 확인하세요: postgresql://user:pass@host:port/db"],
                    severity="critical"
                )
            
            # 연결 테스트
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/') if parsed.path else 'postgres',
                user=parsed.username,
                password=parsed.password
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return CheckResult(
                is_healthy=True,
                message=f"PostgreSQL ({env}): 연결 성공 - {parsed.hostname}",
                details=[f"버전: {version.split(',')[0]}"],
                severity="info"
            )
            
        except ImportError:
            return CheckResult(
                is_healthy=False,
                message=f"PostgreSQL ({env}): psycopg2 패키지 미설치",
                recommendations=["uv add psycopg2-binary"],
                severity="critical"
            )
        except Exception as e:
            suggestion = self._generate_postgres_suggestion(connection_uri, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"PostgreSQL ({env}): 연결 실패",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_redis_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Redis 연결 테스트 (Feature Store online store용).
        
        Args:
            env: 환경 이름
            config: Config 딕셔너리
            
        Returns:
            CheckResult 또는 None (Redis 미설정시)
        """
        fs_config = config.get('feature_store', {}).get('feast_config', {})
        online_store = fs_config.get('online_store', {})
        
        if online_store.get('type') != 'redis':
            return None
            
        connection_string = online_store.get('connection_string')
        if not connection_string:
            return None
            
        try:
            import redis
            
            # Redis 연결 테스트
            r = redis.from_url(f"redis://{connection_string}")
            info = r.info()
            
            return CheckResult(
                is_healthy=True,
                message=f"Redis ({env}): 연결 성공",
                details=[f"버전: {info.get('redis_version', 'Unknown')}"],
                severity="info"
            )
            
        except ImportError:
            return CheckResult(
                is_healthy=False,
                message=f"Redis ({env}): redis 패키지 미설치",
                recommendations=["uv add redis"],
                severity="critical"
            )
        except Exception as e:
            suggestion = self._generate_redis_suggestion(connection_string, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"Redis ({env}): 연결 실패",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_feature_store_config(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Feature Store 설정 검증.
        
        Args:
            env: 환경 이름
            config: Config 딕셔너리
            
        Returns:
            CheckResult 또는 None (Feature Store 미설정시)
        """
        fs_config = config.get('feature_store')
        if not fs_config or fs_config.get('provider') == 'none':
            return None
            
        try:
            feast_config = fs_config.get('feast_config', {})
            required_fields = ['project', 'registry']
            
            missing_fields = [field for field in required_fields if not feast_config.get(field)]
            if missing_fields:
                return CheckResult(
                    is_healthy=False,
                    message=f"Feature Store ({env}): 필수 설정 누락",
                    recommendations=[f"feast_config.{field} 설정 추가 필요" for field in missing_fields],
                    severity="important"
                )
            
            # Registry 경로 검증
            registry_path = feast_config.get('registry')
            if registry_path and not registry_path.startswith(('gs://', 's3://', 'http')):
                registry_file = Path(registry_path)
                if not registry_file.parent.exists():
                    return CheckResult(
                        is_healthy=False,
                        message=f"Feature Store ({env}): Registry 경로 없음",
                        recommendations=[f"mkdir -p {registry_file.parent}"],
                        severity="warning"
                    )
            
            return CheckResult(
                is_healthy=True,
                message=f"Feature Store ({env}): 설정 검증 완료",
                details=[f"Project: {feast_config.get('project')}"],
                severity="info"
            )
            
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                message=f"Feature Store ({env}): 설정 오류",
                recommendations=["feature_store 설정을 확인하세요"],
                severity="important"
            )
    
    def _generate_mlflow_suggestion(self, tracking_uri: str, error: str) -> str:
        """MLflow 연결 오류에 대한 해결책 제시."""
        if "Connection refused" in error:
            if "localhost" in tracking_uri or "127.0.0.1" in tracking_uri:
                return "mlflow ui --host 0.0.0.0 --port 5002"
            return f"MLflow 서버 실행 확인: {tracking_uri}"
        elif "Name or service not known" in error:
            return "네트워크 연결 및 MLflow 서버 주소 확인"
        return f"tracking_uri 설정 확인: {tracking_uri}"
    
    def _generate_postgres_suggestion(self, connection_uri: str, error: str) -> str:
        """PostgreSQL 연결 오류에 대한 해결책 제시."""
        if "Connection refused" in error:
            if "localhost" in connection_uri or "127.0.0.1" in connection_uri:
                return "cd ../mmp-local-dev && docker-compose up -d postgres"
            return "PostgreSQL 서버 실행 확인"
        elif "authentication failed" in error:
            return "DB_USER, DB_PASSWORD 환경변수 확인"
        elif "does not exist" in error:
            return "데이터베이스 생성 또는 연결 정보 확인"
        return f"connection_uri 설정 확인"
    
    def _generate_redis_suggestion(self, connection_string: str, error: str) -> str:
        """Redis 연결 오류에 대한 해결책 제시."""
        if "Connection refused" in error:
            if "localhost" in connection_string or "127.0.0.1" in connection_string:
                return "cd ../mmp-local-dev && docker-compose up -d redis"
            return "Redis 서버 실행 확인"
        return f"Redis 연결 설정 확인: {connection_string}"