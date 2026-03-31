"""
Validate Command Implementation
Recipe + Config YAML 사전 검증 (학습 없이 설정 오류를 조기 발견)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from typing_extensions import Annotated

from mmp.cli.utils import CLIProgress
from mmp.cli.utils.env_loader import load_env_for_config
from mmp.settings import SettingsFactory, ValidationOrchestrator, __version__
from mmp.utils.core.logger import log_config, log_error, log_sys

logger = logging.getLogger(__name__)


def _check_data_source(
    data_path: Optional[str],
    config_adapter_type: str,
) -> Tuple[bool, str, int, int]:
    """
    데이터 소스 접근 가능 여부 확인.

    Returns:
        (success, message, row_count, col_count)
    """
    if not data_path:
        return True, "no data path provided (skipped)", 0, 0

    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".parquet", ".pq"):
        if not path.exists():
            return False, f"file not found: {data_path}", 0, 0

        try:
            import pandas as pd

            if suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)

            rows, cols = df.shape
            return True, f"{data_path} accessible ({rows:,} rows)", rows, cols
        except Exception as e:
            return False, f"failed to read {data_path}: {e}", 0, 0

    # SQL이나 기타 소스는 연결 검증을 하지 않는다 (system-check 영역)
    return True, f"non-file source (adapter: {config_adapter_type})", 0, 0


def _check_feature_columns(
    data_path: Optional[str],
    feature_columns: Optional[List[str]],
) -> Tuple[bool, str, List[str]]:
    """
    데이터 파일에 feature_columns가 존재하는지 확인.

    Returns:
        (success, message, missing_columns)
    """
    if not data_path or not feature_columns:
        return True, "skipped (no data path or no feature_columns defined)", []

    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix not in (".csv", ".parquet", ".pq"):
        return True, "skipped (non-file source)", []

    if not path.exists():
        return True, "skipped (file not found, already reported)", []

    try:
        import pandas as pd

        if suffix == ".csv":
            # 컬럼명만 확인하면 되므로 첫 행만 읽는다
            df = pd.read_csv(path, nrows=1)
        else:
            df = pd.read_parquet(path)

        data_columns = set(df.columns.tolist())
        missing = [col for col in feature_columns if col not in data_columns]

        if missing:
            found = len(feature_columns) - len(missing)
            total = len(feature_columns)
            return (
                False,
                f"{found}/{total} columns found, missing: {missing}",
                missing,
            )

        total = len(feature_columns)
        return True, f"All {total} feature columns found in data", []

    except Exception as e:
        return True, f"skipped (read error: {e})", []


def _format_pydantic_errors(exc: Exception) -> List[str]:
    """Pydantic ValidationError를 읽기 쉬운 메시지 목록으로 변환."""
    lines: List[str] = []
    try:
        for err in exc.errors():  # type: ignore[union-attr]
            loc = " -> ".join(str(loc) for loc in err.get("loc", []))
            msg = err.get("msg", str(err))
            lines.append(f"  {loc}: {msg}")
    except (AttributeError, TypeError):
        lines.append(f"  {exc}")
    return lines


def validate_command(
    recipe_path: Annotated[str, typer.Option("--recipe", "-r", help="Recipe 파일 경로")],
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config 파일 경로")],
    data_path: Annotated[
        Optional[str],
        typer.Option("--data", "-d", help="학습 데이터 파일 경로 (SQL fetcher 사용 시 생략 가능)"),
    ] = None,
    context_params: Annotated[
        Optional[str], typer.Option("--params", "-p", help="Jinja 템플릿 파라미터 (JSON)")
    ] = None,
) -> None:
    """
    Recipe + Config 사전 검증 (학습 없이).

    Recipe와 Config 파일을 로드하고 Pydantic 파싱, Jinja 렌더링,
    Catalog/Business/Compatibility 검증을 수행합니다.
    --data 옵션 시 데이터 파일 접근 및 feature_columns 존재 여부까지 확인합니다.

    Args:
        recipe_path: Recipe YAML 파일 경로
        config_path: Config YAML 파일 경로
        data_path: 학습 데이터 파일 경로
        context_params: 추가 파라미터 (JSON 형식)

    Examples:
        mmp validate -r recipes/model.yaml -c configs/local.yaml
        mmp validate -r recipes/model.yaml -c configs/dev.yaml -d data/train.csv
        mmp validate -r recipes/model.yaml -c configs/prod.yaml --params '{"date": "2024-01-01"}'

    Raises:
        typer.Exit: 검증 실패 시 exit code 1
    """
    verbose = logging.getLogger().level == logging.DEBUG

    # 데이터 관련 스텝은 --data가 주어졌을 때만 포함
    total_steps = 5 if data_path else 3
    progress = CLIProgress(total_steps=total_steps, verbose=verbose)
    progress.header(__version__)

    # 검증 결과를 누적
    results: List[Tuple[str, str, str]] = []  # (status, label, message)
    has_failure = False

    try:
        # 0. 환경변수 로드
        load_env_for_config(config_path)

        # -----------------------------------------------------------
        # Step 1: Parsing Recipe
        # -----------------------------------------------------------
        progress.step_start("Parsing Recipe")
        log_config(f"Recipe: {recipe_path}")

        params: Optional[Dict[str, Any]] = json.loads(context_params) if context_params else None

        # Recipe 파싱은 SettingsFactory 내부에서 수행되지만,
        # 단계별로 결과를 표시하기 위해 factory 인스턴스를 직접 사용한다.
        factory = SettingsFactory()
        try:
            recipe = factory._load_recipe(recipe_path)
        except FileNotFoundError:
            progress.step_fail()
            results.append(("FAIL", "Recipe", f"File not found: {recipe_path}"))
            log_error(f"Recipe 파일을 찾을 수 없습니다: {recipe_path}", "Validate")
            log_sys("Run 'mmp get-recipe' to create a recipe file")
            has_failure = True
            _print_summary(results, progress)
            raise typer.Exit(code=1)
        except ValueError as e:
            progress.step_fail()
            error_lines = _format_pydantic_errors(e)
            results.append(("FAIL", "Recipe", f"Parsing failed:\n{''.join(error_lines)}"))
            log_error(f"Recipe 파싱 실패: {e}", "Validate")
            has_failure = True
            _print_summary(results, progress)
            raise typer.Exit(code=1)

        progress.step_done()
        results.append(("OK", "Recipe", "Valid YAML, all required fields present"))

        # -----------------------------------------------------------
        # Step 2: Parsing Config
        # -----------------------------------------------------------
        progress.step_start("Parsing Config")
        log_config(f"Config: {config_path}")

        try:
            config = factory._load_config(config_path)
        except FileNotFoundError:
            progress.step_fail()
            results.append(("FAIL", "Config", f"File not found: {config_path}"))
            log_error(f"Config 파일을 찾을 수 없습니다: {config_path}", "Validate")
            log_sys("Run 'mmp get-config' to create a config file")
            has_failure = True
            _print_summary(results, progress)
            raise typer.Exit(code=1)
        except ValueError as e:
            progress.step_fail()
            error_lines = _format_pydantic_errors(e)
            results.append(("FAIL", "Config", f"Parsing failed:\n{''.join(error_lines)}"))
            log_error(f"Config 파싱 실패: {e}", "Validate")
            has_failure = True
            _print_summary(results, progress)
            raise typer.Exit(code=1)

        env_name = getattr(getattr(config, "environment", None), "name", "local")
        progress.step_done()
        results.append(("OK", "Config", f"Valid YAML, environment: {env_name}"))

        # data_path가 있으면 recipe에 반영
        if data_path:
            factory._process_data_path(recipe, data_path, params)

        # -----------------------------------------------------------
        # Step 3: Schema validation (catalog, business, compatibility)
        # -----------------------------------------------------------
        progress.step_start("Schema validation")

        orchestrator = ValidationOrchestrator()
        validation_result = orchestrator.validate_for_training(config, recipe)

        if not validation_result.is_valid:
            progress.step_fail()
            results.append(("FAIL", "Validation", validation_result.error_message))
            log_error(f"검증 실패: {validation_result.error_message}", "Validate")
            has_failure = True
            _print_summary(results, progress)
            raise typer.Exit(code=1)

        warning_count = len(validation_result.warnings)
        if warning_count > 0:
            stats = f"passed ({warning_count} warning{'s' if warning_count != 1 else ''})"
        else:
            stats = "(catalog, business, compatibility)"
        progress.step_done(stats)

        # 검증 통과 메시지
        checks_label = "All checks passed"
        if warning_count > 0:
            checks_label += f" ({warning_count} warning{'s' if warning_count != 1 else ''})"
        results.append(("OK", "Validation", checks_label))

        # 경고를 별도 항목으로 추가
        for w in validation_result.warnings:
            results.append(("WARN", "Preprocessor", w))

        # -----------------------------------------------------------
        # Step 4: Data source check (--data가 있을 때만)
        # -----------------------------------------------------------
        if data_path:
            progress.step_start("Data source check")

            adapter_type = getattr(
                getattr(config, "data_source", None), "adapter_type", "unknown"
            )
            ds_ok, ds_msg, row_count, col_count = _check_data_source(data_path, adapter_type)

            if ds_ok:
                if row_count > 0:
                    progress.step_done(f"({row_count:,} rows, {col_count} columns)")
                else:
                    progress.step_done()
                results.append(("OK", "DataSource", ds_msg))
            else:
                progress.step_fail()
                results.append(("FAIL", "DataSource", ds_msg))
                log_error(ds_msg, "Validate")
                has_failure = True

            # -----------------------------------------------------------
            # Step 5: Feature columns check (--data가 있을 때만)
            # -----------------------------------------------------------
            progress.step_start("Feature columns check")

            feature_columns = None
            if hasattr(recipe, "data") and hasattr(recipe.data, "data_interface"):
                feature_columns = getattr(recipe.data.data_interface, "feature_columns", None)

            fc_ok, fc_msg, missing = _check_feature_columns(data_path, feature_columns)

            if fc_ok:
                if feature_columns:
                    progress.step_done(f"({len(feature_columns)}/{len(feature_columns)} columns found)")
                else:
                    progress.step_done("(no feature_columns defined)")
                results.append(("OK", "FeatureColumns", fc_msg))
            else:
                progress.step_fail()
                results.append(("FAIL", "FeatureColumns", fc_msg))
                log_error(f"누락된 컬럼: {missing}", "Validate")
                has_failure = True

        # -----------------------------------------------------------
        # Results Summary
        # -----------------------------------------------------------
        _print_summary(results, progress)

        if has_failure:
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        progress.step_fail()
        log_sys("Validation cancelled by user")
        raise typer.Exit(code=0)
    except FileNotFoundError as e:
        progress.step_fail()
        log_error(f"File not found: {e}", "Validate")
        log_sys("Run 'mmp get-config' or 'mmp get-recipe' to create files")
        raise typer.Exit(code=1)
    except ValueError as e:
        progress.step_fail()
        log_error(f"Configuration error: {e}", "Validate")
        raise typer.Exit(code=1)
    except Exception as e:
        progress.step_fail()
        log_error(str(e), "Validate")
        raise typer.Exit(code=1)


def _print_summary(
    results: List[Tuple[str, str, str]],
    progress: CLIProgress,
) -> None:
    """검증 결과 요약 테이블 출력."""
    status_map = {
        "OK": "[OK]  ",
        "FAIL": "[FAIL]",
        "WARN": "[WARN]",
    }

    sys.stdout.write("\nResults Summary:\n")
    for status, label, message in results:
        tag = status_map.get(status, "[????]")
        sys.stdout.write(f"      {tag} {label:<16} {message}\n")
    sys.stdout.flush()
