"""
System Check Command Implementation
"""

from pathlib import Path
import typer
from typing_extensions import Annotated
import yaml

from rich.console import Console
from src.cli.utils.system_checker import SystemChecker
from src.cli.utils.config_loader import load_environment


def system_check_command(
    config_path: Annotated[
        str, 
        typer.Option("--config-path", "-c", help="체크할 config YAML 파일 경로 (필수)")
    ],
    actionable: Annotated[
        bool, 
        typer.Option("--actionable", "-a", help="실패 시 구체적인 해결책 표시")
    ] = False
) -> None:
    """
    환경 설정 파일 기반으로 시스템 연결 상태를 검사합니다.
    
    지정된 config YAML 파일의 설정을 읽어서 
    실제로 설정된 서비스들의 연결 상태를 검증합니다:
    
    - MLflow tracking server 연결
    - 데이터 어댑터 (PostgreSQL, BigQuery, S3, GCS 등)
    - Feature Store (Feast, Tecton 등)
    - Artifact Storage
    - Serving 설정
    - Monitoring 설정
    
    Examples:
        # 특정 config 파일 체크
        mmp system-check --config-path configs/local.yaml
        mmp system-check --config-path configs/dev.yaml
        
        # 해결책 포함
        mmp system-check --config-path configs/dev.yaml --actionable
    """
    console = Console()
    
    try:
        # 1. Config 파일 경로 검증
        config_file_path = Path(config_path)
        if not config_file_path.exists():
            console.print(f"❌ Config 파일을 찾을 수 없습니다: {config_file_path}", style="red")
            console.print("\n💡 먼저 'mmp get-config'를 실행하여 설정 파일을 생성하세요.", style="yellow")
            raise typer.Exit(1)
        
        # 2. 환경 이름 추출 (환경 변수 로드용)
        env_name = config_file_path.stem  # 파일명에서 확장자 제거
        env_file = Path(f".env.{env_name}")
        
        if env_file.exists():
            try:
                load_environment(env_name)
                console.print(f"✅ 환경 변수 로드: .env.{env_name}", style="green")
            except Exception as e:
                console.print(f"⚠️ 환경 변수 로드 실패: {e}", style="yellow")
        else:
            console.print(f"ℹ️ 환경 변수 파일이 없습니다: .env.{env_name}", style="blue")
        
        # 3. Config 파일 로드
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        console.print(f"✅ Config 로드: {config_file_path}", style="green")
        
        # 4. System Checker 실행
        console.print("\n🔍 시스템 연결 상태를 확인하는 중...\n")
        
        checker = SystemChecker(config, env_name, str(config_file_path))
        results = checker.run_all_checks()
        
        # 5. 결과 표시
        checker.display_results(show_actionable=actionable)
        
    except KeyboardInterrupt:
        console.print("\n⚠️ 시스템 체크가 중단되었습니다.", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ 시스템 체크 중 오류 발생: {e}", style="red")
        raise typer.Exit(1)