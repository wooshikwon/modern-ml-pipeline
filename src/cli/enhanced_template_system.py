"""
M04-1-1 Enhanced Template System
완전한 Jinja2 템플릿 시스템 구현

CLAUDE.md 원칙 준수:
- 구조화된 환경 설정
- Jinja2 템플릿 엔진  
- 단일 진실 공급원 (Single Source of Truth)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal
from jinja2 import Environment, FileSystemLoader, Template
import shutil
import typer

# 환경별 설정 매핑 (계획서 섹션 7.2)
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'local': {
        'adapter': 'storage',
        'hyperparameter_tuning': False,
        'n_estimators': 10,
        'max_depth': 3,
        'n_jobs': 2,
        'data_source': 'data/sample.csv',
        'loader_name': 'local_data_loader',
        'feature_store': False,
        'n_trials': 10
    },
    'dev': {
        'adapter': 'sql', 
        'hyperparameter_tuning': True,
        'n_estimators': 100,
        'max_depth': 10,
        'n_jobs': -1,
        'data_source': 'sql/features.sql',
        'loader_name': 'sql_data_loader',
        'feature_store': True,
        'n_trials': 50
    },
    'prod': {
        'adapter': 'sql',
        'hyperparameter_tuning': True, 
        'n_estimators': 200,
        'max_depth': 15,
        'n_jobs': -1,
        'data_source': 'sql/production_features.sql',
        'loader_name': 'sql_data_loader',
        'feature_store': True,
        'n_trials': 100
    }
}

@dataclass
class TemplateConfig:
    """템플릿 생성을 위한 설정 클래스"""
    environment: Literal['local', 'dev', 'prod']
    recipe_type: str
    target_directory: Path
    additional_features: List[str] = field(default_factory=list)

class EnhancedTemplateGenerator:
    """향상된 템플릿 생성기 - Jinja2 기반"""
    
    def __init__(self, templates_dir: Path):
        """
        템플릿 생성기 초기화
        
        Args:
            templates_dir: 템플릿 파일들이 위치한 디렉토리
        """
        self.templates_dir = templates_dir
        self.recipes_templates_dir = templates_dir / "recipes" / "templates"
        
        # Jinja2 환경 설정
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.recipes_templates_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_project(self, config: TemplateConfig) -> None:
        """
        환경별 맞춤 프로젝트 생성 (계획서 섹션 5.2)
        
        Args:
            config: 템플릿 생성 설정
        """
        self._create_directory_structure(config.target_directory)
        self._generate_config_files(config)
        self._generate_recipes(config)  
        self._copy_static_files(config)
        self._create_data_files(config)
        self._validate_generation(config)
        
        typer.secho(
            f"✅ 성공: '{config.target_directory.resolve()}'에 {config.environment} 환경용 {config.recipe_type} 프로젝트가 생성되었습니다.",
            fg=typer.colors.GREEN,
        )
    
    def _create_directory_structure(self, target_dir: Path) -> None:
        """기본 디렉토리 구조 생성"""
        directories = ['config', 'recipes', 'data']
        
        for dir_name in directories:
            (target_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _generate_config_files(self, config: TemplateConfig) -> None:
        """환경별 config 파일 생성"""
        config_dir = config.target_directory / "config"
        
        # base.yaml 복사
        base_config_src = self.templates_dir / "config" / "base.yaml"
        if base_config_src.exists():
            shutil.copy2(base_config_src, config_dir / "base.yaml")
        
        # 환경별 설정 파일 생성
        env_config_content = self._generate_environment_config(config.environment)
        (config_dir / f"{config.environment}.yaml").write_text(env_config_content)
    
    def _generate_environment_config(self, environment: str) -> str:
        """환경별 설정 내용 생성"""
        config_content = f"""# {environment.upper()} 환경 설정
        environment: {environment}

        # 환경별 특화 설정
        """
        
        env_settings = ENVIRONMENT_CONFIGS[environment]
        
        if environment == "local":
            config_content += f"""
            database:
            adapter: {env_settings['adapter']}
            source: {env_settings['data_source']}

            logging:
            level: INFO
            
            features:
            store_enabled: {str(env_settings['feature_store']).lower()}
            """
        else:  # dev or prod
            config_content += f"""
            database:
            adapter: {env_settings['adapter']}
            connection_string: ${{MMP_{environment.upper()}_DB_URL}}

            logging:
            level: DEBUG

            features:
            store_enabled: {str(env_settings['feature_store']).lower()}
            store_url: ${{MMP_{environment.upper()}_FEATURE_STORE_URL}}
            """
        
        return config_content
    
    def _generate_recipes(self, config: TemplateConfig) -> None:
        """Jinja2 템플릿을 사용한 레시피 생성"""
        try:
            # 템플릿 로드
            template = self.jinja_env.get_template(f"{config.recipe_type}.yaml.j2")
            
            # 환경별 변수 준비
            template_vars = self._prepare_template_variables(config)
            
            # 템플릿 렌더링
            rendered_content = template.render(**template_vars)
            
            # 레시피 파일 생성
            recipe_file = config.target_directory / "recipes" / f"{config.recipe_type}_recipe.yaml"
            recipe_file.write_text(rendered_content)
            
        except Exception as e:
            raise ValueError(f"레시피 템플릿 생성 실패: {e}")
    
    def _prepare_template_variables(self, config: TemplateConfig) -> Dict[str, Any]:
        """템플릿 렌더링을 위한 변수 준비"""
        env_settings = ENVIRONMENT_CONFIGS[config.environment]
        
        return {
            'environment': config.environment,
            'recipe_type': config.recipe_type,
            'recipe_name': f"{config.environment}_{config.recipe_type}_recipe",
            **env_settings  # 환경별 설정 병합
        }
    
    def _copy_static_files(self, config: TemplateConfig) -> None:
        """정적 파일들 복사"""
        # .env.template 복사
        env_template_src = self.templates_dir / ".env.template"
        if env_template_src.exists():
            shutil.copy2(env_template_src, config.target_directory / ".env.template")
    
    def _create_data_files(self, config: TemplateConfig) -> None:
        """환경별 데이터 파일 생성"""
        if config.environment == "local":
            # local 환경용 샘플 데이터 생성
            data_dir = config.target_directory / "data"
            sample_data_content = """user_id,age,income,session_length,page_views,outcome
            1,25,50000,300,10,1
            2,30,60000,250,8,0
            3,35,75000,400,15,1
            4,28,55000,200,5,0
            5,32,65000,350,12,1
            """
            (data_dir / "sample.csv").write_text(sample_data_content)
        else:
            # dev/prod 환경용 SQL 디렉토리 생성
            sql_dir = config.target_directory / "sql"
            sql_dir.mkdir(exist_ok=True)
            
            # 샘플 SQL 파일 생성
            features_sql_content = f"""-- {config.environment.upper()} 환경 Feature 추출 쿼리
            SELECT 
                user_id,
                event_timestamp,
                age,
                income,
                session_length,
                page_views,
                outcome
            FROM {config.environment}_features_table
            WHERE event_timestamp >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY user_id, event_timestamp;
            """
            (sql_dir / "features.sql").write_text(features_sql_content)
    
    def _validate_generation(self, config: TemplateConfig) -> None:
        """생성된 프로젝트 구조 검증"""
        required_files = [
            "config/base.yaml",
            f"config/{config.environment}.yaml",
            f"recipes/{config.recipe_type}_recipe.yaml",
            ".env.template"
        ]
        
        for file_path in required_files:
            full_path = config.target_directory / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"필수 파일이 생성되지 않았습니다: {file_path}")