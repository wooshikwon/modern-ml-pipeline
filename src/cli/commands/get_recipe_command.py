"""
Get-Recipe Command Implementation
"""

import sys
from pathlib import Path

import typer

VERSION = "1.0.0"


def _print_header() -> None:
    """헤더 출력"""
    sys.stdout.write(f"\nmmp v{VERSION}\n\n")
    sys.stdout.write("Get Recipe: Interactive recipe generator\n\n")
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
    from src.cli.utils.interactive_ui import InteractiveUI
    from src.cli.utils.recipe_builder import RecipeBuilder

    ui = InteractiveUI()

    try:
        _print_header()

        ui.show_panel(
            """환경 독립적인 Recipe 생성을 시작합니다.

        Recipe는 환경 설정과 분리되어 있어,
        다양한 환경에서 재사용할 수 있습니다.""",
            title="Recipe Generator",
            style="green",
        )

        builder = RecipeBuilder()
        _print_step("Initialized", "RecipeBuilder ready")

        recipe_data = builder.build_recipe_interactively()

        task = recipe_data.get("task_choice", "N/A")
        model = recipe_data["model"].get("class_path", "N/A")
        _print_step("Interactive flow", f"Task: {task}, Model: {model}")

        recipe_path = builder.create_recipe_file(recipe_data)
        _print_step("Recipe file", str(recipe_path))

        _show_success_message(recipe_path, recipe_data)

    except KeyboardInterrupt:
        sys.stdout.write("\n  [CANCEL] Recipe generation cancelled by user\n")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        _print_error("File not found", str(e))
        raise typer.Exit(1)
    except ValueError as e:
        _print_error("Invalid value", str(e))
        raise typer.Exit(1)
    except Exception as e:
        _print_error("Recipe generation", str(e))
        raise typer.Exit(1)


def _show_success_message(recipe_path: Path, recipe_data: dict) -> None:
    """성공 메시지 표시"""
    library = recipe_data.get("model", {}).get("library", "")
    ml_extras_libraries = ["xgboost", "lightgbm", "catboost"]
    torch_extras_libraries = ["torch", "pytorch"]

    extras_needed = []
    if library.lower() in ml_extras_libraries:
        extras_needed.append("ml-extras")
    if library.lower() in torch_extras_libraries:
        extras_needed.append("torch-extras")

    sys.stdout.write("\nRecipe created successfully!\n\n")
    sys.stdout.write(f"  File: {recipe_path}\n")
    sys.stdout.write(f"  Task: {recipe_data['task_choice']}\n")
    sys.stdout.write(f"  Model: {recipe_data['model']['class_path']}\n")
    sys.stdout.write(f"  Library: {recipe_data['model']['library']}\n")

    sys.stdout.write("\nNext Steps:\n")

    if extras_needed:
        extras_str = ",".join(extras_needed)
        sys.stdout.write(f'  0. Install dependencies: pip install "modern-ml-pipeline[{extras_str}]"\n')

    sys.stdout.write(f"  1. Review recipe: cat {recipe_path}\n")
    sys.stdout.write("  2. Update column names:\n")
    sys.stdout.write("     - target_column: your actual target column\n")
    sys.stdout.write("     - entity_columns: your actual entity columns\n")
    sys.stdout.write(f"  3. Train model: mmp train -r {recipe_path} -c configs/<env>.yaml -d <data>\n")
    sys.stdout.flush()
