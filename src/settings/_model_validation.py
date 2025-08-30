"""
Model Catalog Validation System
Phase 2 Day 4: Pydantic-based model catalog validation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- Pydantic 기반 데이터 검증
- TDD 기반 개발 예정
"""

from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class HyperparameterSpec(BaseModel):
    """
    Individual hyperparameter specification.
    
    Attributes:
        type: Parameter type (int, float, categorical)
        range: Valid range or choices for the parameter
        default: Default value for the parameter
    """
    type: str = Field(..., description="Parameter type: int, float, categorical")
    range: Union[List[Union[int, float]], List[str]] = Field(..., description="Valid range or choices")
    default: Union[int, float, str] = Field(..., description="Default value")


class ModelSpec(BaseModel):
    """
    Complete model specification from catalog.yaml.
    
    Phase 2에서 구현될 예정:
    - Hyperparameter validation
    - Environment compatibility checking
    - Feature requirements validation
    """
    class_path: str = Field(..., description="Full Python class path")
    description: str = Field(..., description="Model description")
    library: str = Field(..., description="ML library name")
    hyperparameters: Dict[str, Dict[str, Any]] = Field(..., description="Hyperparameter specifications")
    supported_tasks: List[str] = Field(..., description="Supported ML tasks")
    feature_requirements: Dict[str, bool] = Field(..., description="Feature type requirements")


class ModelCatalog(BaseModel):
    """
    Complete model catalog with validation capabilities.
    
    Phase 2에서 구현될 예정:
    - catalog.yaml 파싱 및 검증
    - Recipe 호환성 검증
    - Hyperparameter 범위 검증
    """
    models: Dict[str, List[ModelSpec]] = Field(..., description="Models grouped by task")
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[Union[str, Path]] = None) -> 'ModelCatalog':
        """
        Load model catalog from YAML file.
        
        Args:
            yaml_path: Path to catalog.yaml file. If None, uses default location.
            
        Returns:
            Validated ModelCatalog instance
            
        Raises:
            FileNotFoundError: If catalog file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If catalog structure is invalid
        """
        if yaml_path is None:
            # Default path: src/models/catalog.yaml
            current_dir = Path(__file__).parent
            yaml_path = current_dir.parent / "models" / "catalog.yaml"
        else:
            yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Model catalog file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                catalog_data = yaml.safe_load(f)
            
            # Convert to ModelSpec objects
            validated_models = {}
            for task, models in catalog_data.items():
                validated_models[task] = []
                for model_data in models:
                    # Fill in default values for missing fields
                    if 'supported_tasks' not in model_data:
                        model_data['supported_tasks'] = [task]
                    if 'feature_requirements' not in model_data:
                        model_data['feature_requirements'] = {}
                    if 'hyperparameters' not in model_data:
                        model_data['hyperparameters'] = {}
                    
                    model_spec = ModelSpec(**model_data)
                    validated_models[task].append(model_spec)
            
            return cls(models=validated_models)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to validate catalog structure: {e}")
    
    def validate_recipe_compatibility(self, recipe: Dict[str, Any]) -> bool:
        """
        Validate recipe compatibility with catalog specifications.
        
        Args:
            recipe: Recipe dictionary to validate
            
        Returns:
            True if recipe is compatible with catalog
            
        Raises:
            ValueError: If recipe is incompatible
        """
        # Extract model information from recipe
        model_config = recipe.get('model', {})
        if not model_config:
            raise ValueError("Recipe must contain 'model' configuration")
            
        class_path = model_config.get('class_path')
        if not class_path:
            raise ValueError("Model must specify 'class_path'")
        
        # Find model spec in catalog
        model_spec = self.get_model_spec(class_path)
        if model_spec is None:
            raise ValueError(f"Model {class_path} not found in catalog")
        
        # Validate hyperparameters if provided
        recipe_hyperparams = model_config.get('hyperparameters', {})
        if recipe_hyperparams:
            is_valid = self._validate_hyperparameters(model_spec, recipe_hyperparams)
            if not is_valid:
                return False
        
        return True
    
    def _validate_hyperparameters(self, model_spec: ModelSpec, recipe_hyperparams: Dict[str, Any]) -> bool:
        """
        Validate hyperparameters against model specification.
        
        Args:
            model_spec: Model specification from catalog
            recipe_hyperparams: Hyperparameters from recipe
            
        Returns:
            True if hyperparameters are valid
        """
        tunable_params = model_spec.hyperparameters.get('tunable', {})
        
        # Check each hyperparameter in recipe
        for param_name, param_value in recipe_hyperparams.items():
            if param_name in tunable_params:
                param_spec = tunable_params[param_name]
                
                # Validate type and range
                param_type = param_spec.get('type')
                param_range = param_spec.get('range', [])
                
                if param_type == 'int' and isinstance(param_value, int):
                    if len(param_range) == 2 and not (param_range[0] <= param_value <= param_range[1]):
                        return False
                elif param_type == 'float' and isinstance(param_value, (int, float)):
                    if len(param_range) == 2 and not (param_range[0] <= param_value <= param_range[1]):
                        return False
                elif param_type == 'categorical':
                    if param_value not in param_range:
                        return False
            
            # Allow parameters not in catalog (for flexibility)
        
        return True
    
    def get_model_spec(self, class_path: str) -> Optional[ModelSpec]:
        """
        Find model specification by class path.
        
        Args:
            class_path: Full Python class path
            
        Returns:
            ModelSpec if found, None otherwise
        """
        for task_models in self.models.values():
            for model_spec in task_models:
                if model_spec.class_path == class_path:
                    return model_spec
        return None
    
    def _find_model_spec(self, class_path: str) -> Optional[ModelSpec]:
        """
        Internal method to find model specification by class path.
        
        Args:
            class_path: Full Python class path
            
        Returns:
            ModelSpec if found, None otherwise
        """
        return self.get_model_spec(class_path)