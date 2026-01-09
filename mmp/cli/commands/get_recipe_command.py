"""
Get-Recipe Command Implementation
"""

import sys
from pathlib import Path

import typer

from mmp.cli.utils.header import print_command_header
from mmp.utils.core.logger import log_error


def get_recipe_command() -> None:
    """
    환경 독립적인 모델 Recipe 생성.

    대화형 인터페이스를 통해 Task와 모델을 선택하고,
    환경 설정과 독립적인 Recipe 파일을 생성합니다.

    Process:
        1. Recipe 이름 입력
        2. Task 선택 (Classification, Regression, Clustering, Causal)
        3. 모델 선택 (선택한 Task의 사용 가능한 모델들)
        4. 데이터 설정 (source_uri, target_column, entity_schema)
        5. 전처리 및 평가 설정
        6. Recipe 파일 생성 (recipes/{recipe_name}.yaml)

    생성된 Recipe는 config 파일과 함께 사용:
        mmp train -r recipes/model.yaml -c configs/dev.yaml -d data/train.csv

    Raises:
        typer.Exit: 오류 발생 또는 사용자 취소 시
    """
    from mmp.cli.utils.interactive_ui import InteractiveUI
    from mmp.cli.utils.recipe_builder import RecipeBuilder

    ui = InteractiveUI()

    try:
        print_command_header("📋 Get Recipe", "Interactive recipe generator")

        ui.show_panel(
            """환경 독립적인 Recipe 생성을 시작합니다.

        Recipe는 환경 설정과 분리되어 있어,
        다양한 환경에서 재사용할 수 있습니다.""",
            title="Recipe Generator",
            style="green",
        )

        builder = RecipeBuilder()

        recipe_data = builder.build_recipe_interactively()
        recipe_path = builder.create_recipe_file(recipe_data)
        _show_success_message(recipe_path, recipe_data)

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
        log_error(f"Recipe 생성 실패: {e}", "CLI")
        raise typer.Exit(1)


def _show_success_message(recipe_path: Path, recipe_data: dict) -> None:
    """성공 메시지 표시"""
    from mmp.cli.utils.header import print_divider, print_item, print_section

    library = recipe_data.get("model", {}).get("library", "")
    ml_extras_libraries = ["lightgbm", "catboost"]  # xgboost는 core에 포함
    torch_extras_libraries = ["torch", "pytorch"]

    extras_needed = []
    if library.lower() in ml_extras_libraries:
        extras_needed.append("ml-extras")
    if library.lower() in torch_extras_libraries:
        extras_needed.append("torch-extras")

    task = recipe_data["task_choice"]
    preprocessor_steps = recipe_data.get("preprocessor", {})
    if preprocessor_steps:
        preprocessor_steps = preprocessor_steps.get("steps", [])
    else:
        preprocessor_steps = []

    # 구분선
    print_divider()

    # 결과 섹션
    print_section("OK", "Recipe 생성 완료", style="green", newline=False)
    print_item("FILE", str(recipe_path))
    print_item("TASK", task)
    print_item("MODEL", recipe_data["model"]["class_path"])

    # 다음 단계 안내
    print_section("NEXT", "다음 단계", style="blue")

    step_num = 1

    # 추가 의존성이 필요한 경우
    if extras_needed:
        extras_str = ",".join(extras_needed)
        sys.stdout.write(f"  {step_num}. 추가 의존성 설치\n")
        sys.stdout.write(f'     pipx install --force "modern-ml-pipeline[{extras_str}]"\n')
        step_num += 1

    # Recipe 파일 수정 안내
    sys.stdout.write(f"  {step_num}. Recipe 파일 수정 ({recipe_path})\n")

    task_lower = task.lower()
    sys.stdout.write(f"     - entity_columns: 엔티티 식별 컬럼\n")
    if task_lower != "clustering":
        sys.stdout.write(f"     - target_column: 예측 대상 컬럼\n")
    if task_lower == "causal":
        sys.stdout.write(f"     - treatment_column: Treatment 컬럼\n")
    if task_lower == "timeseries":
        sys.stdout.write(f"     - timestamp_column: 시간 컬럼\n")

    has_encoder = any(s.get("type", "").endswith("_encoder") for s in preprocessor_steps)
    if has_encoder:
        sys.stdout.write(f"     - preprocessor.steps[encoder].columns: 범주형 컬럼\n")
    step_num += 1

    # 데이터 준비
    sys.stdout.write(f"  {step_num}. 데이터 준비\n")
    sys.stdout.write(f"     data/ 디렉토리에 .csv 또는 .sql.j2 파일 생성\n")
    step_num += 1

    # 모델 학습
    sys.stdout.write(f"  {step_num}. 모델 학습\n")
    sys.stdout.write(f"     mmp train -r {recipe_path} -c configs/<env>.yaml -d <data>\n")
    sys.stdout.flush()

    # 마무리 구분선
    print_divider()
