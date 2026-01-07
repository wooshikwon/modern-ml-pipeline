"""
Get-Config Command Implementation
"""

import sys
from pathlib import Path
from typing import Optional

import typer

VERSION = "1.0.0"


def _print_header() -> None:
    """헤더 출력"""
    sys.stdout.write(f"\nmmp v{VERSION}\n\n")
    sys.stdout.write("Get Config: Interactive configuration generator\n\n")
    sys.stdout.flush()


def _print_step(step: str, detail: str = "") -> None:
    """단계 완료 출력"""
    if detail:
        sys.stdout.write(f"  [OK] {step}: {detail}\n")
    else:
        sys.stdout.write(f"  [OK] {step}\n")
    sys.stdout.flush()


def _print_error(step: str, error: str) -> None:
    """에러 출력"""
    sys.stdout.write(f"  [FAIL] {step}: {error}\n")
    sys.stdout.flush()


def _print_info(message: str) -> None:
    """정보 출력"""
    sys.stdout.write(f"  {message}\n")
    sys.stdout.flush()


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
    from src.cli.utils.interactive_ui import InteractiveUI

    ui = InteractiveUI()

    try:
        _print_header()

        ui.show_panel(
            """환경별 설정 파일 생성을 시작합니다.

        MLflow, 데이터 소스, Feature Store 등을 선택하여
        환경에 맞는 config 파일을 생성합니다.""",
            title="Config Generator",
            style="green",
        )

        builder = InteractiveConfigBuilder()

        selections = builder.run_interactive_flow(env_name)
        _print_step("설정 완료", f"환경 '{selections['env_name']}'")

        config_path = builder.generate_config_file(selections["env_name"], selections)
        env_template_path = builder.generate_env_template(selections["env_name"], selections)

        _print_step("Config 파일 생성", str(config_path))
        _print_step("Env 템플릿 생성", str(env_template_path))

        _show_completion_message(selections["env_name"], config_path, env_template_path, selections)

    except KeyboardInterrupt:
        sys.stdout.write("\n  [취소] 사용자에 의해 취소됨\n")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        _print_error("파일 없음", str(e))
        raise typer.Exit(1)
    except ValueError as e:
        _print_error("잘못된 값", str(e))
        raise typer.Exit(1)
    except Exception as e:
        _print_error("Config 생성 실패", str(e))
        raise typer.Exit(1)


def _show_completion_message(
    env_name: str, config_path: Path, env_template_path: Path, selections: dict = None
) -> None:
    """완료 메시지 표시"""
    extras_needed = []
    if selections:
        data_source = selections.get("data_source", "")
        feature_store = selections.get("feature_store", "")
        infer_output = selections.get("inference_output_source", "")

        cloud_sources = ["BigQuery", "GCS", "S3"]
        if data_source in cloud_sources or infer_output in cloud_sources:
            extras_needed.append("cloud-extras")

        if feature_store == "Feast":
            extras_needed.append("feature-store")

    sys.stdout.write("\n다음 단계:\n")

    step_num = 1
    if extras_needed:
        extras_str = ",".join(extras_needed)
        sys.stdout.write(f"  {step_num}. 추가 의존성 설치:\n")
        sys.stdout.write(f'     pipx install --force "modern-ml-pipeline[{extras_str}]"\n')
        step_num += 1

    sys.stdout.write(f"  {step_num}. 환경 파일 준비: cp {env_template_path} .env.{env_name}\n")
    sys.stdout.write(f"  {step_num + 1}. 인증 정보 입력: .env.{env_name} 파일 편집\n")
    sys.stdout.write(f"  {step_num + 2}. 연결 테스트: mmp system-check -c {config_path}\n")
    sys.stdout.write(f"  {step_num + 3}. Recipe 생성: mmp get-recipe\n")
    sys.stdout.write(f"  {step_num + 4}. 모델 학습: mmp train -r recipes/<recipe>.yaml -c {config_path} -d <data>\n")
    sys.stdout.flush()
