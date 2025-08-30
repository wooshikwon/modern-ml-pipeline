"""
Template Synchronization System
M04-1-1 계획서 섹션 5.1 구현

root 디렉토리의 실제 파일을 templates/ 디렉토리로 동기화
"""

from pathlib import Path
import shutil
import yaml
from typing import Dict, Any
import typer


def sync_templates_with_root() -> None:
    """
    root 디렉토리의 실제 파일을 템플릿으로 동기화
    계획서 섹션 5.1 구현
    """
    project_root = Path(__file__).parent.parent.parent
    templates_dir = project_root / "src" / "cli" / "project_templates"
    
    typer.echo("🔄 템플릿 동기화를 시작합니다...")
    
    try:
        # 1. config 파일 동기화
        _sync_config_files(project_root, templates_dir)
        
        # 2. 누락된 환경 설정 파일 확인 및 생성
        _ensure_environment_configs(templates_dir)
        
        # 3. 검증
        _validate_template_completeness(templates_dir)
        
        typer.secho("✅ 템플릿 동기화가 완료되었습니다.", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"❌ 템플릿 동기화 실패: {e}", fg=typer.colors.RED)
        raise


def _sync_config_files(project_root: Path, templates_dir: Path) -> None:
    """config 파일들 동기화"""
    root_config = project_root / "config"
    template_config = templates_dir / "config"
    
    if not root_config.exists():
        typer.echo("⚠️ root config 디렉토리가 없습니다. 건너뜁니다.")
        return
    
    # 기존 config 파일들 복사
    for config_file in root_config.glob("*.yaml"):
        dest_file = template_config / config_file.name
        
        if dest_file.exists():
            # 기존 파일과 비교하여 다른 경우에만 업데이트
            if not _files_are_identical(config_file, dest_file):
                shutil.copy2(config_file, dest_file)
                typer.echo(f"📄 업데이트: {config_file.name}")
        else:
            shutil.copy2(config_file, dest_file)
            typer.echo(f"📄 추가: {config_file.name}")


def _ensure_environment_configs(templates_dir: Path) -> None:
    """필수 환경 설정 파일들이 있는지 확인하고 없으면 생성"""
    template_config = templates_dir / "config"
    
    required_configs = {
        "dev.yaml": """# DEV 환경 설정
environment: dev

# 데이터베이스 설정
database:
  adapter: sql
  connection_string: ${MMP_DEV_DB_URL}

# 로깅 설정  
logging:
  level: DEBUG

# Feature Store 설정
features:
  store_enabled: true
  store_url: ${MMP_DEV_FEATURE_STORE_URL}

# MLflow 설정
mlflow:
  tracking_uri: ${MMP_DEV_MLFLOW_URL}
""",
        "prod.yaml": """# PROD 환경 설정
environment: prod

# 데이터베이스 설정
database:
  adapter: sql
  connection_string: ${MMP_PROD_DB_URL}

# 로깅 설정
logging:
  level: INFO

# Feature Store 설정
features:
  store_enabled: true
  store_url: ${MMP_PROD_FEATURE_STORE_URL}

# MLflow 설정
mlflow:
  tracking_uri: ${MMP_PROD_MLFLOW_URL}
"""
    }
    
    for config_name, content in required_configs.items():
        config_file = template_config / config_name
        
        if not config_file.exists():
            config_file.write_text(content)
            typer.echo(f"📄 생성: {config_name}")


def _files_are_identical(file1: Path, file2: Path) -> bool:
    """두 파일이 동일한지 확인"""
    try:
        return file1.read_text() == file2.read_text()
    except Exception:
        return False


def _validate_template_completeness(templates_dir: Path) -> None:
    """템플릿 완성도 검증"""
    required_structure = {
        "config": ["base.yaml", "local.yaml", "dev.yaml", "prod.yaml"],
        "recipes/templates": ["classification.yaml.j2", "regression.yaml.j2", "clustering.yaml.j2", "causal.yaml.j2"],
        "data": ["local_sample.csv"],
        "data/schemas": ["feature_schema.yaml"]
    }
    
    missing_items = []
    
    for dir_path, files in required_structure.items():
        dir_full_path = templates_dir / dir_path
        
        if not dir_full_path.exists():
            missing_items.append(f"디렉토리: {dir_path}")
            continue
            
        for file_name in files:
            file_path = dir_full_path / file_name
            if not file_path.exists():
                missing_items.append(f"파일: {dir_path}/{file_name}")
    
    if missing_items:
        typer.secho("⚠️ 누락된 템플릿 요소들:", fg=typer.colors.YELLOW)
        for item in missing_items:
            typer.echo(f"  - {item}")
    else:
        typer.secho("✅ 모든 템플릿 요소가 완성되었습니다.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    sync_templates_with_root()