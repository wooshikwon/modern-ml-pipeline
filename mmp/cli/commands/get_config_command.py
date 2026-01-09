"""
Get-Config Command Implementation
"""

import sys
from pathlib import Path
from typing import Optional

import typer

from mmp.cli.utils.header import print_command_header
from mmp.utils.core.logger import log_error


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
    from mmp.cli.utils.config_builder import InteractiveConfigBuilder
    from mmp.cli.utils.interactive_ui import InteractiveUI

    ui = InteractiveUI()

    try:
        print_command_header("⚙️ Get Config", "Interactive configuration generator")

        ui.show_panel(
            """환경별 설정 파일 생성을 시작합니다.

        MLflow, 데이터 소스, Feature Store 등을 선택하여
        환경에 맞는 config 파일을 생성합니다.""",
            title="Config Generator",
            style="green",
        )

        builder = InteractiveConfigBuilder()

        selections = builder.run_interactive_flow(env_name)
        config_path = builder.generate_config_file(selections["env_name"], selections)
        env_template_path = builder.generate_env_template(selections["env_name"], selections)

        _show_completion_message(selections["env_name"], config_path, env_template_path, selections)

    except KeyboardInterrupt:
        sys.stdout.write("\n  [취소] 사용자에 의해 취소됨\n")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        log_error(f"파일 없음: {e}", "CLI")
        raise typer.Exit(1)
    except ValueError as e:
        log_error(f"잘못된 값: {e}", "CLI")
        raise typer.Exit(1)
    except Exception as e:
        log_error(f"Config 생성 실패: {e}", "CLI")
        raise typer.Exit(1)


def _show_completion_message(
    env_name: str, config_path: Path, env_template_path: Path, selections: dict = None
) -> None:
    """완료 메시지 표시"""
    from mmp.cli.utils.header import print_divider, print_item, print_section

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

    # 구분선 + 결과 섹션
    print_divider()
    print_section("OK", "Config 생성 완료", style="green", newline=False)
    print_item("ENV", env_name)
    print_item("CONFIG", str(config_path))
    print_item("TEMPLATE", str(env_template_path))

    # 다음 단계 섹션
    print_section("NEXT", "다음 단계", style="blue")

    step_num = 1
    if extras_needed:
        extras_str = ",".join(extras_needed)
        sys.stdout.write(f"  {step_num}. 추가 의존성 설치\n")
        sys.stdout.write(f'     pipx install --force "modern-ml-pipeline[{extras_str}]"\n')
        step_num += 1

    sys.stdout.write(f"  {step_num}. 환경 파일 준비\n")
    sys.stdout.write(f"     cp {env_template_path} .env.{env_name}\n")
    sys.stdout.write(f"  {step_num + 1}. 인증 정보 입력\n")
    sys.stdout.write(f"     .env.{env_name} 파일 편집\n")
    sys.stdout.write(f"  {step_num + 2}. 연결 테스트\n")
    sys.stdout.write(f"     mmp system-check -c {config_path}\n")
    sys.stdout.write(f"  {step_num + 3}. 데이터 준비\n")
    sys.stdout.write(f"     data/ 디렉토리에 .csv 또는 .sql.j2 파일 생성\n")
    sys.stdout.write(f"  {step_num + 4}. Recipe 생성\n")
    sys.stdout.write(f"     mmp get-recipe\n")
    sys.stdout.write(f"  {step_num + 5}. 모델 학습\n")
    sys.stdout.write(f"     mmp train -r recipes/<recipe>.yaml -c {config_path} -d <data>\n")
    sys.stdout.flush()

    # 마무리 구분선
    print_divider()
