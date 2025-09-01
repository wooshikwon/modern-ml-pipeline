"""
System Check Command Implementation  
Phase 3 Day 5-6: Config-based dynamic system validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import typer
from typing_extensions import Annotated

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.cli.utils.system_check_models import CheckResult
from src.cli.utils.env_loader import load_config_with_env
from src.cli.utils.dynamic_service_checker import DynamicServiceChecker


class ConfigBasedSystemChecker:
    """
    Config 파일 내용에 따라 동적으로 시스템 체크하는 클래스.
    
    configs/*.yaml 파일들을 동적으로 로딩하여 실제 설정된 서비스만 체크합니다:
    - MLflow tracking_uri 기반 연결 테스트
    - PostgreSQL connection_uri 기반 연결 테스트  
    - Redis online_store 기반 연결 테스트
    - Feature Store feast_config 기반 설정 검증
    
    실패 시 구체적인 해결책을 제시합니다.
    """
    
    def __init__(self, config_dir: Path = Path("configs")) -> None:
        """
        Initialize config-based system checker.
        
        Args:
            config_dir: Directory containing config YAML files
            
        Raises:
            FileNotFoundError: If config directory doesn't exist
        """
        self.config_dir = config_dir
        self.configs = self._load_all_configs()
        self.console = Console()
    
    def _load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        config 디렉토리의 모든 yaml 파일 동적 로딩.
        
        Returns:
            Dict mapping config file names to their content
            
        Raises:
            FileNotFoundError: If config directory doesn't exist
        """
        configs = {}
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config 디렉토리를 찾을 수 없습니다: {self.config_dir}")
            
        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError(f"Config 디렉토리에 YAML 파일이 없습니다: {self.config_dir}")
        
        for config_file in yaml_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    configs[config_file.stem] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                # YAML 파싱 오류도 결과에 포함
                configs[config_file.stem] = {"_parse_error": str(e)}
        
        return configs
    
    def run_dynamic_checks(self, actionable: bool = False) -> Dict[str, Any]:
        """
        Run dynamic system checks based on config content.
        
        Args:
            actionable: Whether to provide actionable suggestions
            
        Returns:
            Dict with check results summary
            
        Raises:
            ConfigurationError: If config files are invalid
        """
        results = []
        
        for env_name, config in self.configs.items():
            # YAML 파싱 오류 체크
            if "_parse_error" in config:
                results.append(CheckResult(
                    is_healthy=False,
                    message=f"Config Parse Error ({env_name}): YAML 파싱 오류: {config['_parse_error']}",
                    recommendations=[f"configs/{env_name}.yaml 파일의 YAML 구문을 확인하세요"],
                    severity="critical"
                ))
                continue
            
            # 1. MLflow 연결 체크 (tracking_uri 기반)
            mlflow_result = self._check_mlflow_connection(env_name, config)
            if mlflow_result:
                results.append(mlflow_result)
            
            # 2. PostgreSQL 연결 체크 (data_adapters.sql 기반)
            postgres_result = self._check_postgres_connection(env_name, config)
            if postgres_result:
                results.append(postgres_result)
            
            # 3. Redis 연결 체크 (feature_store.online_store 기반) 
            redis_result = self._check_redis_connection(env_name, config)
            if redis_result:
                results.append(redis_result)
            
            # 4. Feature Store 설정 검증 (feast_config 기반)
            fs_result = self._check_feature_store_config(env_name, config)
            if fs_result:
                results.append(fs_result)
        
        # 간단한 요약 생성 (기존 HealthCheckSummary는 복잡하므로 직접 생성)
        passed_count = sum(1 for r in results if r.is_healthy)
        failed_count = len(results) - passed_count
        
        summary = {
            'results': results,
            'overall_healthy': all(r.is_healthy for r in results),
            'total_checks': len(results),
            'passed_checks': passed_count,
            'failed_checks': failed_count
        }
        
        return summary
    
    def _check_mlflow_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Test actual MLflow server connection.
        
        Args:
            env: Environment name
            config: Config dictionary for the environment
            
        Returns:
            CheckResult or None if MLflow not configured
        """
        mlflow_config = config.get('mlflow')
        if not mlflow_config or 'tracking_uri' not in mlflow_config:
            return None  # MLflow 설정이 없으면 체크하지 않음
            
        tracking_uri = mlflow_config['tracking_uri']
        
        try:
            # MLflow 연결 테스트
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # 환경변수 치환 처리
            resolved_uri = self._resolve_environment_variables(tracking_uri)
            
            original_uri = mlflow.get_tracking_uri()
            mlflow.set_tracking_uri(resolved_uri)
            
            # 간단한 연결 테스트
            client = MlflowClient(resolved_uri)
            experiments = client.search_experiments(max_results=1)
            
            # 원래 URI 복원
            mlflow.set_tracking_uri(original_uri)
            
            return CheckResult(
                is_healthy=True,
                message=f"MLflow Connection ({env}): 연결 성공 - {resolved_uri}",
                details=[f"발견된 실험 수: {len(experiments) if experiments else 0}"],
                severity="info"
            )
            
        except Exception as e:
            suggestion = self._generate_mlflow_suggestion(tracking_uri, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"MLflow Connection ({env}): 연결 실패 - {e}",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_postgres_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Test PostgreSQL database connection.
        
        Args:
            env: Environment name
            config: Config dictionary for the environment
            
        Returns:
            CheckResult or None if PostgreSQL not configured
        """
        # data_adapters.adapters.sql.config.connection_uri 경로로 접근
        adapters = config.get('data_adapters', {}).get('adapters', {})
        sql_adapter = adapters.get('sql', {})
        sql_config = sql_adapter.get('config', {})
        connection_uri = sql_config.get('connection_uri')
        
        if not connection_uri:
            return None  # SQL adapter 설정이 없으면 체크하지 않음
            
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            # connection_uri 파싱
            parsed = urlparse(connection_uri)
            
            # PostgreSQL 연결 테스트
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password
            )
            
            # 간단한 쿼리 테스트
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return CheckResult(
                is_healthy=True,
                message=f"PostgreSQL Connection ({env}): 연결 성공 - {parsed.hostname}:{parsed.port}",
                details=[f"버전: {version[:50]}..."],
                severity="info"
            )
            
        except Exception as e:
            suggestion = self._generate_postgres_suggestion(connection_uri, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"PostgreSQL Connection ({env}): 연결 실패 - {e}",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_redis_connection(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Test Redis connection for Feature Store online store.
        
        Args:
            env: Environment name
            config: Config dictionary for the environment
            
        Returns:
            CheckResult or None if Redis not configured
        """
        fs_config = config.get('feature_store', {}).get('feast_config', {})
        online_store = fs_config.get('online_store', {})
        
        if online_store.get('type') != 'redis':
            return None  # Redis 설정이 아니면 체크하지 않음
            
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
                message=f"Redis Connection ({env}): 연결 성공 - {connection_string}",
                details=[f"Redis 버전: {info.get('redis_version', 'Unknown')}"],
                severity="info"
            )
            
        except Exception as e:
            suggestion = self._generate_redis_suggestion(connection_string, str(e))
            return CheckResult(
                is_healthy=False,
                message=f"Redis Connection ({env}): 연결 실패 - {e}",
                recommendations=[suggestion],
                severity="important"
            )
    
    def _check_feature_store_config(self, env: str, config: Dict[str, Any]) -> Optional[CheckResult]:
        """
        Feature Store 설정 검증.
        
        Args:
            env: Environment name
            config: Config dictionary for the environment
            
        Returns:
            CheckResult or None if Feature Store not configured
        """
        fs_config = config.get('feature_store')
        if not fs_config or fs_config.get('provider') == 'none':
            return None  # Feature Store 미사용 시 체크하지 않음
            
        try:
            feast_config = fs_config.get('feast_config', {})
            required_fields = ['project', 'registry']
            
            missing_fields = [field for field in required_fields if not feast_config.get(field)]
            if missing_fields:
                return CheckResult(
                    is_healthy=False,
                    message=f"Feature Store Config ({env}): 필수 Feast 설정 누락 - {missing_fields}",
                    recommendations=[f"configs/{env}.yaml에서 feature_store.feast_config.{missing_fields[0]} 설정을 추가하세요"],
                    severity="important"
                )
            
            # Feast repo 경로 검증 (registry 파일 기준)
            registry_path = feast_config.get('registry')
            if registry_path and not registry_path.startswith(('gs://', 's3://', 'http')):
                # 로컬 파일 경로인 경우 존재 여부 체크
                registry_file = Path(registry_path)
                if not registry_file.parent.exists():
                    return CheckResult(
                        is_healthy=False,
                        message=f"Feature Store Config ({env}): Feast registry 디렉토리가 존재하지 않습니다 - {registry_file.parent}",
                        recommendations=[f"mkdir -p {registry_file.parent} 또는 올바른 registry 경로를 설정하세요"],
                        severity="warning"
                    )
            
            return CheckResult(
                is_healthy=True,
                message=f"Feature Store Config ({env}): 설정 검증 완료 - {feast_config.get('project')}",
                details=[f"Provider: {fs_config.get('provider')}", f"Registry: {registry_path}"],
                severity="info"
            )
            
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                message=f"Feature Store Config ({env}): 설정 검증 실패 - {e}",
                recommendations=["feature_store 설정을 확인하고 필수 필드를 추가하세요"],
                severity="important"
            )
    
    def _generate_mlflow_suggestion(self, tracking_uri: str, error: str) -> str:
        """
        MLflow 연결 오류에 대한 구체적 해결책.
        
        Args:
            tracking_uri: MLflow tracking URI
            error: Error message
            
        Returns:
            Specific suggestion for the error
        """
        if "Connection refused" in error:
            if "127.0.0.1" in tracking_uri or "localhost" in tracking_uri:
                return "MLflow 서버를 시작하세요: mlflow ui --host 0.0.0.0 --port 5002"
            else:
                return f"MLflow 서버가 실행 중인지 확인하세요: {tracking_uri}"
        elif "Name or service not known" in error:
            return "네트워크 연결을 확인하고 MLflow 서버 주소가 올바른지 검증하세요"
        else:
            return f"tracking_uri 설정을 확인하세요: {tracking_uri}"
    
    def _generate_postgres_suggestion(self, connection_uri: str, error: str) -> str:
        """
        PostgreSQL 연결 오류에 대한 구체적 해결책.
        
        Args:
            connection_uri: PostgreSQL connection URI
            error: Error message
            
        Returns:
            Specific suggestion for the error
        """
        if "Connection refused" in error:
            if "127.0.0.1" in connection_uri or "localhost" in connection_uri:
                return "mmp-local-dev PostgreSQL을 시작하세요: cd ../mmp-local-dev && docker-compose up -d postgres"
            else:
                return "PostgreSQL 서버가 실행 중인지 확인하세요"
        elif "authentication failed" in error:
            return "데이터베이스 사용자명과 비밀번호를 확인하세요"
        elif "database" in error and "does not exist" in error:
            return "데이터베이스가 존재하지 않습니다. 데이터베이스를 생성하거나 연결 정보를 확인하세요"
        else:
            return f"connection_uri 설정을 확인하세요: {connection_uri}"
    
    def _generate_redis_suggestion(self, connection_string: str, error: str) -> str:
        """
        Redis 연결 오류에 대한 구체적 해결책.
        
        Args:
            connection_string: Redis connection string
            error: Error message
            
        Returns:
            Specific suggestion for the error
        """
        if "Connection refused" in error:
            if "localhost" in connection_string or "127.0.0.1" in connection_string:
                return "mmp-local-dev Redis를 시작하세요: cd ../mmp-local-dev && docker-compose up -d redis"
            else:
                return "Redis 서버가 실행 중인지 확인하세요"
        else:
            return f"Redis 연결 설정을 확인하세요: {connection_string}"
    
    def _resolve_environment_variables(self, value: str) -> str:
        """
        환경변수를 치환합니다.
        
        ${VAR_NAME:default_value} 형식을 지원합니다.
        
        Args:
            value: 치환할 문자열
            
        Returns:
            환경변수가 치환된 문자열
        """
        import os
        import re
        
        # ${VAR_NAME:default_value} 패턴 매칭
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match):
            env_expr = match.group(1)
            if ':' in env_expr:
                var_name, default_value = env_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                return os.getenv(env_expr.strip(), match.group(0))
        
        return re.sub(pattern, replace_env_var, value)


# CLI 명령어 구현

def system_check_command(
    env_name: Annotated[
        str,
        typer.Option("--env-name", "-e", help="검사할 환경 이름 (필수)")
    ],
    actionable: bool = typer.Option(False, "--actionable", "-a", help="실행 가능한 해결책 제시")
) -> None:
    """
    특정 환경의 시스템 연결 상태 검사.
    
    지정된 환경의 config를 기반으로 서비스 연결을 테스트합니다:
    - MLflow tracking_uri 기반 연결 테스트
    - PostgreSQL connection_uri 기반 연결 테스트  
    - Redis online_store 기반 연결 테스트
    - Feature Store feast_config 기반 설정 검증
    
    Args:
        env_name: 환경 이름 (필수)
        actionable: 실행 가능한 해결책 제시 여부
    """
    try:
        # v2.0: env_name은 필수 파라미터
        
        if actionable:
            typer.echo(f"🔍 환경 '{env_name}'의 연결 상태를 검사합니다... (실행 가능한 해결책 모드)\n")
        else:
            typer.echo(f"🔍 환경 '{env_name}'의 연결 상태를 검사합니다...\n")
        
        # 환경별 config 로드
        config = load_config_with_env(env_name)
        
        # 단일 환경 체크를 위한 새로운 체커 생성
        checker = DynamicServiceChecker()
        summary = checker.check_single_environment(env_name, config, actionable=actionable)
        
        if actionable:
            # ActionableReporter 사용
            _display_actionable_summary(summary, checker.actionable_reporter)
            
            if not summary['overall_healthy']:
                typer.echo("\n" + "=" * 60)
                typer.secho("💡 실행 가능한 해결 명령어:", fg=typer.colors.GREEN, bold=True)
                
                for result in summary['results']:
                    if not result.is_healthy and result.recommendations:
                        typer.echo(f"🔧 {result.message.split(':')[0]}:")
                        for rec in result.recommendations:
                            typer.echo(f"   {rec}")
        else:
            # 기본 보고서
            _display_basic_summary(summary)
        
        # 전체 상태에 따른 종료 코드
        if not summary['overall_healthy']:
            raise typer.Exit(code=1)
            
    except FileNotFoundError as e:
        typer.secho(f"❌ 설정 파일 오류: {e}", fg=typer.colors.RED)
        typer.echo("\n💡 해결방법:")
        typer.echo("   1. modern-ml-pipeline init 으로 프로젝트를 초기화하세요")
        typer.echo("   2. 또는 configs/ 디렉토리에 YAML 설정 파일이 있는지 확인하세요")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"❌ 시스템 체크 중 오류 발생: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _display_basic_summary(summary: Dict[str, Any]) -> None:
    """
    기본 요약 표시 (Rich 기반).
    
    Args:
        summary: System check summary dictionary
    """
    console = Console()
    passed_count = summary['passed_checks']
    failed_count = summary['failed_checks']
    total_count = summary['total_checks']
    
    # 요약 테이블
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("✅ 성공:", f"[green]{passed_count}[/green]")
    summary_table.add_row("❌ 실패:", f"[red]{failed_count}[/red]") 
    summary_table.add_row("📋 총 검사:", f"[blue]{total_count}[/blue]")
    
    console.print(Panel(summary_table, title="📊 검사 결과 요약"))
    
    if summary['overall_healthy']:
        console.print("\n🎉 [green bold]모든 시스템이 정상적으로 연결되었습니다![/green bold]")
    else:
        console.print(f"\n⚠️ [yellow]{failed_count}개의 연결 문제가 발견되었습니다.[/yellow]")
        
        # 실패 결과 테이블
        if failed_count > 0:
            error_table = Table(show_header=True, header_style="red bold")
            error_table.add_column("서비스", style="cyan")
            error_table.add_column("문제", style="red")
            error_table.add_column("해결책", style="yellow")
            
            for result in summary['results']:
                if not result.is_healthy:
                    service = result.message.split(':')[0]
                    problem = result.message.split(':', 1)[1].strip()
                    solution = result.recommendations[0] if result.recommendations else "해결책 없음"
                    error_table.add_row(service, problem, solution)
            
            console.print(error_table)
        
        console.print("\n💡 [green]더 구체적인 해결방법:[/green]")
        console.print("   [blue]modern-ml-pipeline system-check --actionable[/blue]")


def _display_actionable_summary(summary: Dict[str, Any], reporter: Any) -> None:
    """
    실행 가능한 요약 표시.
    
    Args:
        summary: System check summary dictionary  
        reporter: ActionableReporter instance
    """
    # ActionableReporter의 display 메서드 사용
    # 기존 health 모듈의 구조에 맞춰 간단히 구현
    passed_count = summary['passed_checks']
    failed_count = summary['failed_checks']
    
    typer.echo("📊 액션 가능한 검사 결과:")
    typer.echo(f"   ✅ 성공: {passed_count}")
    typer.echo(f"   ❌ 실패: {failed_count}")
    
    # 성공한 항목들
    if passed_count > 0:
        typer.secho("\n✅ 정상 연결된 서비스:", fg=typer.colors.GREEN, bold=True)
        for result in summary['results']:
            if result.is_healthy:
                typer.echo(f"   • {result.message}")
                if result.details:
                    for detail in result.details:
                        typer.echo(f"     - {detail}")
    
    # 실패한 항목들과 해결책
    if failed_count > 0:
        typer.secho("\n❌ 연결 실패한 서비스:", fg=typer.colors.RED, bold=True)
        for result in summary['results']:
            if not result.is_healthy:
                typer.echo(f"   • {result.message}")
                if result.recommendations:
                    for rec in result.recommendations:
                        typer.echo(f"     💡 {rec}")