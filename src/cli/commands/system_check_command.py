"""
System Check Command Implementation v2.0
Refactored version using modular architecture following Single Responsibility Principle.

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from pathlib import Path
from typing import Dict, Any, Optional
import typer
from typing_extensions import Annotated

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.cli.system_check.manager import DynamicServiceChecker
from src.cli.system_check.models import CheckResult
from src.cli.utils.config_loader import load_environment, load_config_with_env


def system_check_command(
    env_name: Annotated[
        Optional[str], 
        typer.Option("--env-name", "-e", help="특정 환경만 체크 (미지정시 전체)")
    ] = None,
    actionable: Annotated[
        bool, 
        typer.Option("--actionable", "-a", help="실패 시 구체적인 해결책 표시")
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="JSON 형식으로 출력")
    ] = False
) -> None:
    """
    현재 config 파일 기반으로 시스템 연결 상태를 검사합니다.
    
    configs/*.yaml 파일들을 자동으로 읽어서 설정된 서비스들을 체크합니다:
    - MLflow tracking server 연결
    - PostgreSQL 데이터베이스 연결
    - Redis 서버 연결
    - Feature Store 설정 검증
    
    Examples:
        # 모든 환경 체크
        mmp system-check
        
        # 특정 환경만 체크
        mmp system-check --env-name dev
        
        # 해결책 포함
        mmp system-check --actionable
        
        # JSON 출력
        mmp system-check --json
    """
    console = Console()
    
    try:
        # Load environment if specified
        if env_name:
            try:
                load_environment(env_name)
            except FileNotFoundError:
                console.print(f"⚠️  환경 파일 .env.{env_name}을 찾을 수 없습니다.", style="yellow")
                # Continue anyway - config might not need env vars
        
        # Load configuration and run checks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("시스템 체크 중...", total=None)
            
            # Load configuration
            if env_name:
                # Check specific environment
                try:
                    config = load_config_with_env(env_name)
                except FileNotFoundError as e:
                    console.print(f"❌ 환경 '{env_name}'의 설정을 찾을 수 없습니다: {e}", style="red")
                    raise typer.Exit(1)
                    
                # Create checker and run
                checker = DynamicServiceChecker()
                check_results = checker.run_checks(config)
                
                results = {
                    'environments': {
                        env_name: {
                            'checks': [_convert_check_result(r) for r in check_results],
                            'summary': checker.get_summary_stats(check_results)
                        }
                    },
                    'summary': {
                        'all_healthy': all(r.is_healthy for r in check_results),
                        'environments_checked': 1,
                        'total_checks': len(check_results),
                        'healthy': sum(1 for r in check_results if r.is_healthy),
                        'unhealthy': sum(1 for r in check_results if not r.is_healthy),
                        'overall_health_percentage': checker.get_summary_stats(check_results).get('success_rate', 0)
                    }
                }
            else:
                # Check all environments
                from pathlib import Path
                config_dir = Path("configs")
                
                if not config_dir.exists():
                    console.print("❌ configs 디렉토리를 찾을 수 없습니다", style="red")
                    raise typer.Exit(1)
                    
                yaml_files = list(config_dir.glob("*.yaml"))
                if not yaml_files:
                    console.print("❌ configs 디렉토리에 YAML 파일이 없습니다", style="red")
                    raise typer.Exit(1)
                    
                all_results = {}
                total_checks = 0
                total_healthy = 0
                
                for yaml_file in yaml_files:
                    env = yaml_file.stem
                    try:
                        config = load_config_with_env(env)
                        checker = DynamicServiceChecker()
                        check_results = checker.run_checks(config)
                        
                        all_results[env] = {
                            'checks': [_convert_check_result(r) for r in check_results],
                            'summary': checker.get_summary_stats(check_results)
                        }
                        
                        total_checks += len(check_results)
                        total_healthy += sum(1 for r in check_results if r.is_healthy)
                        
                    except Exception as e:
                        console.print(f"⚠️  환경 '{env}' 체크 실패: {e}", style="yellow")
                        
                results = {
                    'environments': all_results,
                    'summary': {
                        'all_healthy': total_healthy == total_checks and total_checks > 0,
                        'environments_checked': len(all_results),
                        'total_checks': total_checks,
                        'healthy': total_healthy,
                        'unhealthy': total_checks - total_healthy,
                        'overall_health_percentage': (total_healthy / total_checks * 100) if total_checks > 0 else 0
                    }
                }
            
            progress.update(task, completed=True)
        
        # Output results
        if json_output:
            import json
            console.print_json(json.dumps(results, indent=2, default=str))
        else:
            _display_results(results, actionable, console)
            
    except KeyboardInterrupt:
        console.print("\n⚠️  시스템 체크가 중단되었습니다.", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ 시스템 체크 중 오류 발생: {e}", style="red")
        raise typer.Exit(1)


def _display_results(results: Dict[str, Any], actionable: bool, console: Console) -> None:
    """
    Display check results in a formatted table.
    
    Args:
        results: Check results from coordinator
        actionable: Whether to show actionable suggestions
        console: Rich console for output
    """
    if 'error' in results:
        console.print(f"❌ {results['error']}", style="red")
        if 'available_environments' in results:
            console.print(f"사용 가능한 환경: {', '.join(results['available_environments'])}")
        return
    
    environments = results.get('environments', {})
    summary = results.get('summary', {})
    
    # Display results for each environment
    for env_name, env_results in environments.items():
        _display_environment_results(env_name, env_results, actionable, console)
    
    # Display overall summary
    _display_summary(summary, console)


def _convert_check_result(result: CheckResult) -> Dict[str, Any]:
    """
    Convert CheckResult to dictionary for display.
    
    Args:
        result: CheckResult object
        
    Returns:
        Dictionary with check result data
    """
    return {
        'service': result.service_name,
        'status': 'healthy' if result.is_healthy else 'unhealthy',
        'message': result.message,
        'details': {
            'suggestion': '\n'.join(result.recommendations) if result.recommendations else None,
            'severity': result.severity
        }
    }


def _display_environment_results(env_name: str, env_results: Dict[str, Any], actionable: bool, console: Console) -> None:
    """
    Display results for a single environment.
    
    Args:
        env_name: Environment name
        env_results: Check results for this environment
        actionable: Whether to show suggestions
        console: Rich console for output
    """
    console.print(f"\n🔍 환경: [bold cyan]{env_name}[/bold cyan]")
    
    checks = env_results.get('checks', [])
    env_summary = env_results.get('summary', {})
    
    if not checks:
        console.print("  설정된 서비스가 없습니다.", style="dim")
        return
    
    # Create results table
    table = Table(title=f"시스템 체크 결과 - {env_name}", show_header=True, header_style="bold magenta")
    table.add_column("서비스", style="cyan", no_wrap=True)
    table.add_column("상태", justify="center")
    table.add_column("메시지", style="dim")
    
    if actionable:
        table.add_column("해결책", style="yellow")
    
    # Add rows for each check
    for check in checks:
        service = check['service']
        status = check['status']
        message = check['message']
        details = check.get('details', {})
        
        # Status emoji and color
        if status == 'healthy':
            status_display = "[green]✅ 정상[/green]"
        elif status == 'warning':
            status_display = "[yellow]⚠️  경고[/yellow]"
        elif status == 'unhealthy':
            status_display = "[red]❌ 실패[/red]"
        else:
            status_display = "[dim]ℹ️  정보[/dim]"
        
        # Build row
        row = [service, status_display, message]
        
        if actionable and 'suggestion' in details:
            row.append(details['suggestion'])
        
        table.add_row(*row)
    
    console.print(table)
    
    # Display environment summary
    health_pct = env_summary.get('health_percentage', 0)
    health_color = "green" if health_pct >= 80 else "yellow" if health_pct >= 50 else "red"
    
    console.print(
        f"  환경 상태: [{health_color}]{health_pct:.1f}% 정상[/{health_color}] "
        f"(정상: {env_summary.get('healthy', 0)}, "
        f"경고: {env_summary.get('warning', 0)}, "
        f"실패: {env_summary.get('unhealthy', 0)})"
    )


def _display_summary(summary: Dict[str, Any], console: Console) -> None:
    """
    Display overall summary across all environments.
    
    Args:
        summary: Overall summary from coordinator
        console: Rich console for output
    """
    console.print("\n" + "="*50)
    
    # Overall health status
    all_healthy = summary.get('all_healthy', False)
    health_pct = summary.get('overall_health_percentage', 0)
    
    if all_healthy:
        console.print("✅ [bold green]모든 시스템이 정상입니다![/bold green]")
    else:
        health_color = "green" if health_pct >= 80 else "yellow" if health_pct >= 50 else "red"
        console.print(f"📊 전체 시스템 상태: [{health_color}]{health_pct:.1f}% 정상[/{health_color}]")
    
    # Statistics
    panel_content = (
        f"검사한 환경: {summary.get('environments_checked', 0)}개\n"
        f"전체 체크: {summary.get('total_checks', 0)}개\n"
        f"✅ 정상: {summary.get('healthy', 0)}개\n"
        f"⚠️  경고: {summary.get('warning', 0)}개\n"
        f"❌ 실패: {summary.get('unhealthy', 0)}개"
    )
    
    console.print(Panel(panel_content, title="요약", border_style="cyan"))
    
    # List unhealthy services
    unhealthy = summary.get('unhealthy_services', [])
    if unhealthy:
        console.print("\n❌ 실패한 서비스:", style="red")
        for service in unhealthy:
            console.print(f"  - {service}")
    
    warning = summary.get('warning_services', [])
    if warning:
        console.print("\n⚠️  경고 서비스:", style="yellow")
        for service in warning:
            console.print(f"  - {service}")
    
    # Next steps
    if not all_healthy:
        console.print("\n💡 다음 단계:", style="cyan")
        console.print("  1. --actionable 옵션으로 구체적인 해결책 확인")
        console.print("  2. 실패한 서비스의 설정 확인")
        console.print("  3. 필요한 서비스 시작 (Docker, 로컬 서버 등)")