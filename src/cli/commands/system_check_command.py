"""
System Check Command Implementation
"""

import logging
from pathlib import Path
from typing import Dict

import typer
import yaml
from typing_extensions import Annotated

from src.cli.utils.cli_progress import CLIProgress
from src.cli.utils.system_checker import CheckResult, CheckStatus, SystemChecker

logger = logging.getLogger(__name__)


def load_environment(env_name: str) -> None:
    """환경변수 파일 로드"""
    from pathlib import Path

    from dotenv import load_dotenv

    env_file = Path.cwd() / f".env.{env_name}"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    else:
        raise FileNotFoundError(f".env.{env_name} 파일을 찾을 수 없습니다.")


def system_check_command(
    config_path: Annotated[
        str, typer.Option("--config", "-c", help="체크할 config YAML 파일 경로 (필수)")
    ],
    recipe_path: Annotated[
        str, typer.Option("--recipe", "-r", help="체크할 recipe YAML 파일 경로 (선택)")
    ] = None,
    actionable: Annotated[
        bool, typer.Option("--actionable", "-a", help="실패 시 구체적인 해결책 표시")
    ] = False,
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
        mmp system-check --config configs/local.yaml
        mmp system-check --config configs/dev.yaml

        # 해결책 포함
        mmp system-check --config configs/dev.yaml --actionable
    """
    try:
        # 1. Config 파일 경로 검증
        config_file_path = Path(config_path)
        env_name = config_file_path.stem

        progress = CLIProgress(total_steps=5)
        progress.header("1.0.0")

        # Step 1: Loading configuration
        progress.step_start("Loading configuration")
        if not config_file_path.exists():
            progress.step_fail()
            logger.error(f"      [FAIL] Config file not found: {config_file_path}")
            logger.error("      Run 'mmp get-config' to create a config file")
            raise typer.Exit(1)

        # 환경 변수 로드
        env_file = Path(f".env.{env_name}")
        if env_file.exists():
            try:
                load_environment(env_name)
            except Exception as e:
                logger.warning(f"      [WARN] Environment failed ({e})")
        
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        recipe = None
        if recipe_path:
            recipe_file_path = Path(recipe_path)
            if recipe_file_path.exists():
                with open(recipe_file_path, "r", encoding="utf-8") as f:
                    recipe = yaml.safe_load(f)
        
        progress.step_done()

        # SystemChecker 초기화 (로직 수행용)
        checker = SystemChecker(config, env_name, str(config_file_path), recipe=recipe)
        results: Dict[str, CheckResult] = {}

        # Step 2: Checking package deps
        progress.step_start("Checking package deps")
        pkg_res = checker.check_package_dependencies()
        results[pkg_res.service] = pkg_res
        if pkg_res.status == CheckStatus.SUCCESS:
            progress.step_done("verified")
        else:
            progress.step_fail()

        # Step 3: Verifying MLflow
        progress.step_start("Verifying MLflow")
        if "mlflow" in config:
            ml_res = checker.check_mlflow()
            results[ml_res.service] = ml_res
            if ml_res.status == CheckStatus.SUCCESS:
                uri = config["mlflow"].get("tracking_uri", "./mlruns")
                progress.step_done(f"connected: {uri}")
            else:
                progress.step_fail()
        else:
            progress.step_skip("not configured")

        # Step 4: Validating data source
        progress.step_start("Validating data source")
        if "data_source" in config:
            ds_res = checker.check_data_source(config["data_source"])
            results[ds_res.service] = ds_res
            
            # 호환성 체크 포함
            comp_failed = False
            if recipe:
                comp_res = checker.check_data_source_compatibility()
                results[comp_res.service] = comp_res
                if comp_res.status == CheckStatus.FAILED:
                    comp_failed = True
                
            if ds_res.status == CheckStatus.SUCCESS and not comp_failed:
                adapter = config["data_source"].get("adapter_type", "unknown")
                progress.step_done(f"ok ({adapter})")
            else:
                progress.step_fail()
        else:
            progress.step_skip("no data source")

        # Step 5: Checking other services
        progress.step_start("Checking other services")
        # Feature Store
        if "feature_store" in config:
            provider = config["feature_store"].get("provider", "").lower()
            if provider and provider != "none":
                fs_res = checker.check_feature_store()
                results[fs_res.service] = fs_res
        
        # Serving
        if config.get("serving", {}).get("enabled", False):
            sv_res = checker.check_serving()
            results[sv_res.service] = sv_res

        progress.step_done()

        # 결과 요약 테이블 출력
        logger.info("\nResults Summary:")
        success_count = sum(1 for r in results.values() if r.status == CheckStatus.SUCCESS)
        failed_count = sum(1 for r in results.values() if r.status == CheckStatus.FAILED)
        warning_count = sum(1 for r in results.values() if r.status == CheckStatus.WARNING)
        skipped_count = sum(1 for r in results.values() if r.status == CheckStatus.SKIPPED)

        for service, res in results.items():
            status_tag = {
                CheckStatus.SUCCESS: "[OK]  ",
                CheckStatus.FAILED: "[FAIL]",
                CheckStatus.WARNING: "[WARN]",
                CheckStatus.SKIPPED: "[SKIP]",
            }.get(res.status, "[FAIL]")

            logger.info(f"      {status_tag} {service:<20} {res.message}")

        summary_text = f"{success_count} passed, {failed_count} failed, {warning_count} warnings"
        if failed_count == 0:
            if warning_count == 0:
                logger.info(f"\nAll system checks passed! ({summary_text})")
            else:
                logger.info(f"\nSystem is working with warnings. ({summary_text})")
        else:
            logger.error(f"\n[ERROR] {failed_count} check(s) failed. ({summary_text})")

        # 해결책(Solutions) 출력
        if actionable and (failed_count > 0 or warning_count > 0):
            logger.info("\nSolutions:")
            for service, res in results.items():
                if res.status in [CheckStatus.FAILED, CheckStatus.WARNING] and res.solution:
                    indented_sol = res.solution.replace("\n", "\n" + " " * 21)
                    logger.info(f"      {service:<15}: {indented_sol}")

        if failed_count > 0:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        logger.info("\nSystem check cancelled by user")
        raise typer.Exit(0)
    except FileNotFoundError as e:
        progress.step_fail()
        logger.error(f"  File not found: {e}")
        logger.error("  Run 'mmp get-config' to create a config file")
        raise typer.Exit(1)
    except ValueError as e:
        progress.step_fail()
        logger.error(f"  Configuration error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        progress.step_fail()
        logger.error(f"  {e}")
        raise typer.Exit(1)
