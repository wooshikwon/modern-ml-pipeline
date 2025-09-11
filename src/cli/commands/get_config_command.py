"""
Get-Config Command Implementation
"""

from typing import Optional
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_config_command(
    env_name: Optional[str] = typer.Option(None, "--env-name", "-e", help="환경 이름")
) -> None:
    """
    대화형으로 환경별 설정 파일을 생성합니다.
    
    MLflow, 데이터 소스, Feature Store, Artifact Storage 등을 
    대화형으로 선택하여 환경별 config YAML과 .env 템플릿을 생성합니다.
    
    Examples:
        mmp get-config
        mmp get-config --env-name dev
        mmp get-config --env-name production
        
    생성되는 파일:
        - configs/{env_name}.yaml: 환경 설정 파일
        - .env.{env_name}.template: 환경 변수 템플릿
    """
    from src.cli.utils.config_builder import InteractiveConfigBuilder
    
    try:
        builder = InteractiveConfigBuilder()
        
        # 대화형 플로우 실행
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