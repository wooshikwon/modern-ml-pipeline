"""
Get-Config Command Implementation
Phase 1: Interactive configuration generation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Optional, Dict, Any
import typer
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_config_command(
    env_name: Optional[str] = typer.Option(None, "--env-name", "-e", help="환경 이름"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="비대화형 모드"),
    template: Optional[str] = typer.Option(None, "--template", "-t", help="템플릿 사용 (local/dev/prod)")
) -> None:
    """
    대화형으로 환경별 설정 파일을 생성합니다.
    
    환경별 config YAML과 .env 템플릿을 생성합니다.
    
    Examples:
        mmp get-config
        mmp get-config --env-name dev
        mmp get-config --template local --non-interactive
    """
    from src.cli.utils.config_builder import InteractiveConfigBuilder
    
    try:
        builder = InteractiveConfigBuilder()
        
        if non_interactive and template:
            # 템플릿 기반 빠른 생성
            _create_from_template(env_name or "local", template)
            return
        
        # 대화형 플로우
        selections = builder.run_interactive_flow(env_name)
        
        # 파일 생성
        config_path = builder.generate_config_file(selections['env_name'], selections)
        env_template_path = builder.generate_env_template(selections['env_name'], selections)
        
        # 완료 메시지
        _show_completion_message(selections['env_name'], config_path, env_template_path)
        
    except KeyboardInterrupt:
        console.print("\n❌ 설정 생성이 취소되었습니다.", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ 오류 발생: {e}", style="red")
        raise typer.Exit(1)


def _create_from_template(env_name: str, template: str) -> None:
    """
    템플릿 기반으로 빠르게 설정 생성.
    
    Args:
        env_name: 환경 이름
        template: 템플릿 이름 (local/dev/prod)
    """
    from src.cli.utils.config_builder import InteractiveConfigBuilder
    
    builder = InteractiveConfigBuilder()
    
    # 템플릿별 기본 설정
    template_configs = {
        'local': {
            'env_name': env_name,
            'project_name': 'ml-pipeline',
            'data_source': 'postgresql',
            'db_host': 'localhost',
            'db_port': '5432',
            'db_name': 'mlflow',
            'db_user': 'postgres',
            'mlflow_type': 'local',
            'mlflow_uri': './mlruns',
            'feature_store_enabled': False,
            'storage_type': 'local',
            'storage_path': './data'
        },
        'dev': {
            'env_name': env_name,
            'project_name': 'ml-pipeline',
            'data_source': 'postgresql',
            'db_host': 'dev-db.example.com',
            'db_port': '5432',
            'db_name': 'mlflow_dev',
            'db_user': 'developer',
            'mlflow_type': 'remote',
            'mlflow_uri': 'http://mlflow-dev.example.com:5000',
            'feature_store_enabled': True,
            'feature_store_type': 'redis',
            'redis_host': 'dev-redis.example.com',
            'redis_port': '6379',
            'storage_type': 'gcs',
            'gcs_bucket': 'dev-ml-artifacts'
        },
        'prod': {
            'env_name': env_name,
            'project_name': 'ml-pipeline',
            'data_source': 'bigquery',
            'bq_project': 'your-project',
            'bq_dataset': 'ml_data',
            'mlflow_type': 'remote',
            'mlflow_uri': 'http://mlflow-prod.example.com:5000',
            'feature_store_enabled': True,
            'feature_store_type': 'redis',
            'redis_host': 'prod-redis.example.com',
            'redis_port': '6379',
            'storage_type': 'gcs',
            'gcs_bucket': 'prod-ml-artifacts'
        }
    }
    
    if template not in template_configs:
        console.print(f"❌ 알 수 없는 템플릿: {template}", style="red")
        console.print("사용 가능한 템플릿: local, dev, prod", style="yellow")
        raise typer.Exit(1)
    
    selections = template_configs[template]
    selections['env_name'] = env_name  # Override with provided env_name
    
    # 파일 생성
    config_path = builder.generate_config_file(env_name, selections)
    env_template_path = builder.generate_env_template(env_name, selections)
    
    _show_completion_message(env_name, config_path, env_template_path)


def _show_completion_message(env_name: str, config_path: Path, env_template_path: Path) -> None:
    """
    완료 메시지 표시.
    
    Args:
        env_name: 환경 이름
        config_path: 생성된 config 파일 경로
        env_template_path: 생성된 .env 템플릿 경로
    """
    console.print("\n✅ [bold green]설정 파일 생성 완료![/bold green]")
    console.print(f"  📄 Config: {config_path}")
    console.print(f"  📄 Env Template: {env_template_path}")
    
    # 다음 단계 안내
    next_steps = Panel.fit(
        f"""💡 [bold cyan]다음 단계:[/bold cyan]
        
1. 환경 변수 파일 준비:
   [cyan]cp {env_template_path} .env.{env_name}[/cyan]
   
2. .env.{env_name} 파일을 편집하여 실제 인증 정보 입력

3. 시스템 연결 테스트:
   [cyan]mmp system-check --env-name {env_name}[/cyan]
   
4. Recipe 생성:
   [cyan]mmp get-recipe[/cyan]
   
5. 학습 실행:
   [cyan]mmp train --recipe-file recipes/model.yaml --env-name {env_name}[/cyan]
""",
        title="다음 단계",
        border_style="cyan"
    )
    
    console.print(next_steps)