"""
System Check Command Implementation
"""

from pathlib import Path
import typer
from typing_extensions import Annotated
import yaml

from src.cli.utils.system_checker import SystemChecker
from src.utils.core.console import (
    cli_command_start, cli_command_success, cli_command_error,
    cli_system_check_header, cli_validation_summary,
    cli_step_start, cli_step_complete, get_rich_console
)


# config_loader.py 제거 - 간단한 load_environment 직접 구현
def load_environment(env_name: str) -> None:
    """환경변수 파일 로드 (config_loader.py 대체)"""
    from dotenv import load_dotenv
    from pathlib import Path

    env_file = Path.cwd() / f".env.{env_name}"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    else:
        raise FileNotFoundError(f".env.{env_name} 파일을 찾을 수 없습니다.")


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
    try:
        # 1. Config 파일 경로 검증
        config_file_path = Path(config_path)
        env_name = config_file_path.stem  # 파일명에서 확장자 제거

        cli_system_check_header(config_path, env_name)

        if not config_file_path.exists():
            cli_command_error("System Check",
                            f"Config 파일을 찾을 수 없습니다: {config_file_path}",
                            "'mmp get-config'를 실행하여 설정 파일을 생성하세요")
            raise typer.Exit(1)

        # 2. 환경 변수 로드
        cli_step_start("Config 파일 검증")
        env_file = Path(f".env.{env_name}")

        if env_file.exists():
            try:
                load_environment(env_name)
                cli_step_complete("환경 변수 로드", f".env.{env_name}")
            except Exception as e:
                cli_step_complete("환경 변수 로드", f"실패: {e}")
        else:
            cli_step_complete("환경 변수", f"파일 없음: .env.{env_name}")

        # 3. Config 파일 로드
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        cli_step_complete("Config 로드", str(config_file_path))

        # 4. System Checker 실행을 시각적으로 표시
        console = get_rich_console()
        with console.progress_tracker("system_check", 5, "시스템 연결 검사") as update:
            checker = SystemChecker(config, env_name, str(config_file_path))
            results = checker.run_all_checks()  # update 진행률은 SystemChecker 내부에서 처리
            update(5)  # 검사 완료

        # 5. 결과 표시 - console_manager 형식으로 변환
        validation_results = []
        all_passed = True

        for component, result in results.items():
            status_map = {
                'success': 'pass',
                'error': 'fail',
                'warning': 'warning'
            }
            status = status_map.get(result.get('status', 'error'), 'fail')

            if status != 'pass':
                all_passed = False

            validation_results.append({
                "item": component,
                "status": status,
                "details": result.get('message', '')
            })

        cli_validation_summary(validation_results, "시스템 연결 검사 결과")

        # SystemChecker의 display_results도 호출 (actionable 옵션 지원)
        if actionable:
            checker.display_results(show_actionable=True)

        if all_passed:
            cli_command_success("System Check", ["모든 시스템 구성 요소 검사 완료"])

    except KeyboardInterrupt:
        cli_command_error("System Check", "사용자에 의해 중단됨")
        raise typer.Exit(1)
    except Exception as e:
        cli_command_error("System Check", f"시스템 체크 중 오류 발생: {e}")
        raise typer.Exit(1)