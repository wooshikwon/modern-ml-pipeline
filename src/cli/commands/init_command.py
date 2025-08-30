"""
Init Command Implementation
Phase 5 Day 10: Simplified project initialization with mmp-local-dev integration

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader


def create_project_structure(project_path: Path, with_mmp_dev: bool = False) -> None:
    """
    Create mmp-local-dev compatible project structure.
    
    Args:
        project_path: Target directory for project creation
        with_mmp_dev: Whether to generate mmp-local-dev compatible config
    """
    # 1. 기본 디렉토리 생성
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "config").mkdir(exist_ok=True)
    (project_path / "recipes").mkdir(exist_ok=True)
    (project_path / "data").mkdir(exist_ok=True)
    (project_path / "docs").mkdir(exist_ok=True)
    
    # 2. Config 파일들 생성
    _generate_config_files(project_path, with_mmp_dev)
    
    # 3. 샘플 데이터 생성
    _generate_sample_data(project_path)
    
    # 4. 프로젝트 문서 생성
    _generate_project_docs(project_path)


def clone_mmp_local_dev(parent_dir: Path) -> None:
    """
    Clone mmp-local-dev repository to parent directory.
    
    Args:
        parent_dir: Parent directory where mmp-local-dev will be cloned
    
    Raises:
        subprocess.CalledProcessError: If git clone fails
    """
    mmp_dev_path = parent_dir / "mmp-local-dev"
    
    if mmp_dev_path.exists():
        return  # 이미 존재하면 skip
    
    # Git clone 실행
    cmd = [
        "git", "clone", 
        "https://github.com/your-org/mmp-local-dev.git",
        str(mmp_dev_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, 
            cmd, 
            output=result.stdout, 
            stderr=result.stderr
        )


def _generate_config_files(project_path: Path, with_mmp_dev: bool) -> None:
    """Generate environment-specific config files using Jinja2 templates."""
    
    # 템플릿 디렉토리 설정
    templates_dir = Path(__file__).parent.parent / "templates" / "configs"
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    
    # Jinja2 환경 설정
    env = Environment(loader=FileSystemLoader(templates_dir))
    
    # 공통 템플릿 변수
    template_vars = {
        'project_name': project_path.name,
        'with_mmp_dev': with_mmp_dev,
        'timestamp': datetime.now().isoformat(),
        'experiment_name': f"{project_path.name}_experiment",
        
        # 환경별 변수들
        'serving_enabled': 'false',
        'serving_host': '0.0.0.0',
        'serving_port': 8000,
        'enable_hyperparameter_tuning': 'false',
        'tuning_timeout': 300,
        'tuning_n_jobs': 2,
        'enable_monitoring': 'true',
        'enable_logging': 'true',
        'enable_postgres_storage': 'true' if with_mmp_dev else 'false',
        
        # mmp-local-dev 특화 변수들
        'dev_db_uri': 'postgresql://postgres:postgres@127.0.0.1:5432/mlflow' if with_mmp_dev else '${DEV_DB_URI}',
        'dev_db_host': 'localhost' if with_mmp_dev else '${DEV_DB_HOST}',
        'dev_db_port': 5432,
        'dev_db_name': 'mlflow' if with_mmp_dev else '${DEV_DB_NAME}',
        'dev_db_user': 'postgres' if with_mmp_dev else '${DEV_DB_USER}',
        'dev_db_password': 'postgres' if with_mmp_dev else '${DEV_DB_PASSWORD}',
        'dev_redis_uri': 'localhost:6379' if with_mmp_dev else '${DEV_REDIS_URI}',
        
        # 운영 환경 변수들
        'prod_gcp_project': '${PROD_GCP_PROJECT}',
        'prod_bucket': '${PROD_BUCKET}',
        'prod_db_uri': '${PROD_DB_URI}',
        'prod_mlflow_uri': '${PROD_MLFLOW_URI}',
        'prod_redis_uri': '${PROD_REDIS_URI}',
    }
    
    # 각 환경별 config 템플릿 렌더링
    configs = ['base', 'local', 'dev', 'prod']
    for config_name in configs:
        try:
            template = env.get_template(f"{config_name}.yaml.j2")
            rendered = template.render(**template_vars)
            
            config_path = project_path / "config" / f"{config_name}.yaml"
            config_path.write_text(rendered, encoding='utf-8')
            
        except Exception as e:
            raise RuntimeError(f"Failed to render {config_name}.yaml template: {e}") from e


def _generate_sample_data(project_path: Path) -> None:
    """Generate sample CSV data for local development."""
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(2, 1.5, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # CSV 저장
    sample_path = project_path / "data" / "sample_data.csv"
    df.to_csv(sample_path, index=False)


def _generate_project_docs(project_path: Path) -> None:
    """Generate project-specific documentation."""
    
    readme_content = f"""# {project_path.name}

이 프로젝트는 Modern ML Pipeline을 사용하여 생성되었습니다.

## 시작하기

1. 레시피 생성:
```bash
modern-ml-pipeline get-recipe
```

2. 모델 훈련:
```bash
modern-ml-pipeline train --recipe-file recipes/<generated_recipe>.yaml
```

3. 시스템 상태 확인:
```bash
modern-ml-pipeline system-check
```

## 프로젝트 구조

```
{project_path.name}/
├── config/          # 환경별 설정 파일
├── recipes/         # ML 레시피 파일들
├── data/           # 샘플 데이터 및 데이터 파일들
└── docs/           # 프로젝트 문서
```

## 다음 단계

- `modern-ml-pipeline get-recipe`로 첫 번째 ML 레시피를 생성해보세요
- `config/` 디렉토리의 설정을 프로젝트에 맞게 수정하세요
- `data/sample_data.csv`를 참고하여 실제 데이터를 준비하세요
"""
    
    readme_path = project_path / "docs" / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')