"""
System Check Command Implementation
Phase 5: Simplified config-based validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 환경 기반 검증
"""

from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
import yaml

from rich.console import Console
from src.cli.utils.system_checker import SystemChecker
from src.cli.utils.config_loader import load_environment


def system_check_command(
    env_name: Annotated[
        str, 
        typer.Option("--env-name", "-e", help="체크할 환경 이름 (필수)")
    ],
    actionable: Annotated[
        bool, 
        typer.Option("--actionable", "-a", help="실패 시 구체적인 해결책 표시")
    ] = False
) -> None:
    """
    환경 설정 파일 기반으로 시스템 연결 상태를 검사합니다.
    
    configs/{env_name}.yaml 파일의 설정을 읽어서 
    실제로 설정된 서비스들의 연결 상태를 검증합니다:
    
    - MLflow tracking server 연결
    - 데이터 어댑터 (PostgreSQL, BigQuery, S3, GCS 등)
    - Feature Store (Feast, Tecton 등)
    - Artifact Storage
    - Serving 설정
    - Monitoring 설정
    
    Examples:
        # 특정 환경 체크
        mmp system-check --env-name local
        mmp system-check --env-name dev
        
        # 해결책 포함
        mmp system-check --env-name dev --actionable
    """
    console = Console()
    
    try:
        # 1. 환경 변수 로드 (있는 경우)
        env_file = Path(f".env.{env_name}")
        if env_file.exists():
            try:
                load_environment(env_name)
                console.print(f"✅ 환경 변수 로드: .env.{env_name}", style="green")
            except Exception as e:
                console.print(f"⚠️ 환경 변수 로드 실패: {e}", style="yellow")
        else:
            console.print(f"ℹ️ 환경 변수 파일이 없습니다: .env.{env_name}", style="blue")
        
        # 2. Config 파일 로드
        config_path = Path("configs") / f"{env_name}.yaml"
        if not config_path.exists():
            console.print(f"❌ Config 파일을 찾을 수 없습니다: {config_path}", style="red")
            console.print("\n💡 먼저 'mmp get-config'를 실행하여 설정 파일을 생성하세요.", style="yellow")
            raise typer.Exit(1)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        console.print(f"✅ Config 로드: {config_path}", style="green")
        
        # 3. System Checker 실행
        console.print("\n🔍 시스템 연결 상태를 확인하는 중...\n")
        
        checker = SystemChecker(config, env_name)
        results = checker.run_all_checks()
        
        # 4. 결과 표시
        checker.display_results(show_actionable=actionable)
        
    except KeyboardInterrupt:
        console.print("\n⚠️ 시스템 체크가 중단되었습니다.", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ 시스템 체크 중 오류 발생: {e}", style="red")
        raise typer.Exit(1)