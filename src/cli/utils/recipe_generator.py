"""
Catalog 기반 Recipe 생성기
src/models/catalog/ 정보와 환경별 설정을 조합하여 최적화된 recipe.yaml 생성
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from src.settings._model_validation import ModelSpec
from .environment_configs import ENVIRONMENT_CONFIGS
from .evaluation_metrics import get_task_metrics, get_tuning_config


class CatalogBasedRecipeGenerator:
    """Catalog 정보 기반 Recipe YAML 생성기."""
    
    def __init__(self) -> None:
        """Initialize the recipe generator."""
        self.console = Console()
        
        # Jinja2 환경 설정
        template_dir = Path(__file__).parent.parent / "templates" / "recipes"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_recipe(
        self, 
        task_type: str, 
        model_spec: ModelSpec
    ) -> Path:
        """
        Catalog 정보와 환경 설정을 조합하여 Recipe 생성.
        
        Args:
            environment: 타겟 환경 (local, dev, prod)
            task_type: ML 태스크 타입
            model_spec: Catalog에서 선택된 모델 스펙
            
        Returns:
            Path: 생성된 레시피 파일 경로
            
        Raises:
            ValueError: If invalid parameters or template rendering fails
            IOError: If file creation fails
        """
        try:
            # 1. 템플릿 변수 준비
            template_vars = self._prepare_template_variables(environment, task_type, model_spec)
            
            # 2. 레시피 파일명 생성
            recipe_filename = self._generate_recipe_filename(environment, task_type, model_spec)
            
            # 3. recipes 디렉토리 확인/생성
            recipes_dir = Path("recipes")
            recipes_dir.mkdir(exist_ok=True)
            recipe_path = recipes_dir / recipe_filename
            
            # 4. Jinja2 템플릿 렌더링
            template = self.jinja_env.get_template("base_recipe.yaml.j2")
            rendered_content = template.render(**template_vars)
            
            # 5. 파일 저장
            recipe_path.write_text(rendered_content, encoding="utf-8")
            
            self.console.print(f"[green]✅ 레시피 생성 완료: {recipe_path}[/green]")
            return recipe_path
            
        except Exception as e:
            self.console.print(f"[red]❌ 레시피 생성 실패: {e}[/red]")
            raise ValueError(f"Recipe generation failed: {e}")
    
    def _prepare_template_variables(
        self, 
        environment: str, 
        task_type: str, 
        model_spec: ModelSpec
    ) -> Dict[str, Any]:
        """템플릿 렌더링을 위한 변수들 준비."""
        
        # 환경별 기본 설정 가져오기
        env_config = ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS['local'])
        
        # 태스크별 평가 메트릭 가져오기
        task_metrics = get_task_metrics(task_type)
        tuning_config = get_tuning_config(task_type)
        
        # 모델의 하이퍼파라미터 처리
        hyperparameters = self._process_model_hyperparameters(model_spec, environment)
        
        # 튜닝 가능한 파라미터 추출
        tunable_params = self._extract_tunable_parameters(model_spec)
        
        # 레시피명 생성
        model_name = model_spec.class_path.split('.')[-1].lower()
        recipe_name = f"{environment}_{task_type.lower()}_{model_name}"
        
        return {
            'recipe_name': recipe_name,
            'environment': environment,
            'task_type': task_type,
            'model_info': {
                'class_path': model_spec.class_path,
                'library': model_spec.library,
                'description': getattr(model_spec, 'description', f'{model_name} model')
            },
            'hyperparameters': hyperparameters,
            'environment_config': env_config,
            'data_config': env_config['data_config'],
            'evaluation_metrics': task_metrics,
            'tuning_config': tuning_config,
            'tunable_params': tunable_params,
            'validation_config': env_config['validation_config'],
            'generation_date': datetime.now().isoformat()
        }
    
    def _process_model_hyperparameters(self, model_spec: ModelSpec, environment: str) -> Dict[str, Any]:
        """모델의 하이퍼파라미터를 환경에 맞게 처리."""
        hyperparams = {}
        
        if hasattr(model_spec, 'hyperparameters') and model_spec.hyperparameters:
            # Fixed parameters (모든 환경에서 동일)
            if 'fixed' in model_spec.hyperparameters:
                hyperparams.update(model_spec.hyperparameters['fixed'])
            
            # Environment-specific defaults
            if 'environment_defaults' in model_spec.hyperparameters:
                env_defaults = model_spec.hyperparameters['environment_defaults'].get(environment, {})
                hyperparams.update(env_defaults)
        
        return hyperparams
    
    def _extract_tunable_parameters(self, model_spec: ModelSpec) -> Dict[str, Any]:
        """튜닝 가능한 하이퍼파라미터 추출."""
        tunable_params = {}
        
        if (hasattr(model_spec, 'hyperparameters') and 
            model_spec.hyperparameters and 
            'tunable' in model_spec.hyperparameters):
            tunable_params = model_spec.hyperparameters['tunable']
        
        return tunable_params
    
    def _generate_recipe_filename(self, environment: str, task_type: str, model_spec: ModelSpec) -> str:
        """레시피 파일명 생성."""
        model_name = model_spec.class_path.split('.')[-1].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{environment}_{task_type.lower()}_{model_name}_{timestamp}.yaml"