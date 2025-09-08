"""
Model Catalog Validator (v3.0)
모델 카탈로그 YAML 검증 및 Settings 검증 도구
CLI 구조와 완벽 호환 - 완전히 재작성됨
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Literal
from pathlib import Path
import yaml
import importlib

from src.utils.system.logger import logger


class TunableParameter(BaseModel):
    """튜닝 가능한 파라미터 정의 (모델 카탈로그용)"""
    type: Literal["int", "float", "categorical"] = Field(..., description="파라미터 타입")
    range: Union[List[Any], Dict[str, Any]] = Field(..., description="파라미터 범위 또는 선택지")
    default: Optional[Any] = Field(None, description="기본값")
    log: bool = Field(False, description="로그 스케일 사용 여부 (Optuna용)")
    
    @validator('range')
    def validate_range(cls, v, values):
        """범위 타입 검증"""
        param_type = values.get('type')
        
        if param_type in ['int', 'float']:
            # 숫자형은 [min, max] 형태
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError(f"{param_type} 타입은 [min, max] 형태여야 합니다")
            if v[0] >= v[1]:
                raise ValueError(f"범위가 잘못되었습니다: {v[0]} >= {v[1]}")
        elif param_type == 'categorical':
            # 범주형은 리스트
            if not isinstance(v, list) or len(v) < 2:
                raise ValueError("categorical 타입은 2개 이상의 선택지가 필요합니다")
        
        return v


class HyperparameterSpec(BaseModel):
    """
    하이퍼파라미터 스펙 (모델 카탈로그용)
    Recipe의 hyperparameters 구조로 변환 가능
    """
    fixed: Dict[str, Any] = Field(default_factory=dict, description="고정 파라미터")
    tunable: Dict[str, TunableParameter] = Field(default_factory=dict, description="튜닝 가능 파라미터")
    
    def to_recipe_hyperparameters(self, enable_tuning: bool = False) -> Dict[str, Any]:
        """
        Recipe용 하이퍼파라미터 구조로 변환
        CLI recipe.yaml 구조와 호환
        
        Args:
            enable_tuning: 튜닝 활성화 여부
            
        Returns:
            Recipe hyperparameters 딕셔너리
        """
        if enable_tuning:
            # 튜닝 활성화 구조
            return {
                "tuning_enabled": True,
                "fixed": self.fixed,
                "tunable": {
                    name: {
                        "type": param.type,
                        "range": param.range,
                        "log": param.log  # Optuna용 추가 정보
                    }
                    for name, param in self.tunable.items()
                }
            }
        else:
            # 튜닝 비활성화 구조 (기본값 사용)
            values = self.fixed.copy()
            
            # Tunable 파라미터의 기본값 추가
            for name, param in self.tunable.items():
                if param.default is not None:
                    values[name] = param.default
                elif isinstance(param.range, list) and param.range:
                    # 기본값이 없으면 범위의 첫 번째 값 사용
                    if param.type == 'categorical':
                        values[name] = param.range[0]
                    else:  # int, float
                        # 범위의 중간값 사용
                        if param.type == 'int':
                            values[name] = (param.range[0] + param.range[1]) // 2
                        else:  # float
                            values[name] = (param.range[0] + param.range[1]) / 2
            
            return {
                "tuning_enabled": False,
                "values": values
            }


class ModelSpec(BaseModel):
    """모델 스펙 정의 (모델 카탈로그 YAML)"""
    class_path: str = Field(..., description="모델 클래스 전체 경로")
    library: str = Field(..., description="라이브러리 (sklearn, xgboost 등)")
    description: str = Field("", description="모델 설명")
    hyperparameters: HyperparameterSpec = Field(
        default_factory=HyperparameterSpec,
        description="하이퍼파라미터 스펙"
    )
    supported_tasks: List[str] = Field(default_factory=list, description="지원 태스크 목록")
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ModelSpec":
        """YAML 파일에서 모델 스펙 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # hyperparameters가 없으면 빈 객체 생성
        if 'hyperparameters' not in data:
            data['hyperparameters'] = {}
        
        # tunable 파라미터를 TunableParameter 객체로 변환
        if 'tunable' in data.get('hyperparameters', {}):
            tunable = {}
            for name, spec in data['hyperparameters']['tunable'].items():
                tunable[name] = TunableParameter(**spec)
            data['hyperparameters']['tunable'] = tunable
        
        return cls(**data)
    
    def is_compatible_with_task(self, task: str) -> bool:
        """특정 태스크와 호환되는지 확인"""
        if not self.supported_tasks:
            # 지원 태스크가 명시되지 않으면 모든 태스크 지원 가정
            return True
        
        return task.lower() in [t.lower() for t in self.supported_tasks]


class ModelCatalog(BaseModel):
    """모델 카탈로그 관리자"""
    models: Dict[str, Dict[str, ModelSpec]] = Field(
        default_factory=dict,
        description="태스크별 모델 스펙 (task -> model_name -> spec)"
    )
    
    @classmethod
    def load_from_directory(cls, catalog_dir: Path) -> "ModelCatalog":
        """
        디렉토리에서 모델 카탈로그 로드
        
        구조:
        catalog_dir/
        ├── Classification/
        │   ├── RandomForest.yaml
        │   └── XGBoost.yaml
        ├── Regression/
        │   └── LinearRegression.yaml
        └── ...
        
        Args:
            catalog_dir: 모델 카탈로그 디렉토리 경로
            
        Returns:
            ModelCatalog 객체
        """
        catalog = cls()
        
        if not catalog_dir.exists():
            logger.warning(f"모델 카탈로그 디렉토리가 없습니다: {catalog_dir}")
            return catalog
        
        # 태스크별 디렉토리 순회
        for task_dir in catalog_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            # DeepLearning 태스크는 제외 (CLI에서 지원 안 함)
            if task_dir.name == "DeepLearning":
                continue
            
            task_name = task_dir.name
            catalog.models[task_name] = {}
            
            # 모델 YAML 파일 로드
            for model_file in task_dir.glob("*.yaml"):
                model_name = model_file.stem
                
                try:
                    spec = ModelSpec.from_yaml(model_file)
                    catalog.models[task_name][model_name] = spec
                    logger.debug(f"모델 로드: {task_name}/{model_name}")
                except Exception as e:
                    logger.warning(f"모델 스펙 로드 실패 ({model_file}): {str(e)}")
        
        logger.info(f"모델 카탈로그 로드 완료: {len(catalog.models)} 태스크")
        return catalog
    
    def get_model_spec(self, task: str, model_name: str) -> Optional[ModelSpec]:
        """
        특정 모델 스펙 조회
        
        Args:
            task: 태스크 이름
            model_name: 모델 이름
            
        Returns:
            ModelSpec 또는 None
        """
        task_models = self.models.get(task, {})
        
        # 정확한 이름 매칭 시도
        if model_name in task_models:
            return task_models[model_name]
        
        # 대소문자 무시 매칭 시도
        for name, spec in task_models.items():
            if name.lower() == model_name.lower():
                return spec
        
        return None
    
    def list_models_for_task(self, task: str) -> List[str]:
        """특정 태스크의 모델 목록"""
        return list(self.models.get(task, {}).keys())
    
    def list_tasks(self) -> List[str]:
        """사용 가능한 태스크 목록"""
        return list(self.models.keys())


class Validator:
    """
    Settings 검증기
    Config와 Recipe의 유효성을 검증
    """
    
    # 태스크별 유효한 메트릭
    VALID_METRICS = {
        "classification": [
            "accuracy", "precision", "recall", "f1", "f1_score", "roc_auc", 
            "log_loss", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"
        ],
        "regression": [
            "mae", "mse", "rmse", "r2", "r2_score", "mape", 
            "explained_variance", "max_error", "mean_absolute_error",
            "mean_squared_error", "mean_absolute_percentage_error"
        ],
        "clustering": [
            "silhouette_score", "davies_bouldin", "calinski_harabasz",
            "inertia", "adjusted_rand_score", "mutual_info_score"
        ],
        "causal": [
            "ate", "att", "confidence_intervals", "propensity_score",
            "treatment_effect", "heterogeneous_effect"
        ]
    }
    
    def __init__(self, catalog_dir: Optional[Path] = None):
        """
        Validator 초기화
        
        Args:
            catalog_dir: 모델 카탈로그 디렉토리 (선택사항)
        """
        if catalog_dir:
            self.catalog = ModelCatalog.load_from_directory(catalog_dir)
        else:
            self.catalog = None
    
    def validate_recipe(self, recipe) -> List[str]:
        """
        Recipe 검증
        
        Args:
            recipe: 검증할 Recipe 객체
            
        Returns:
            오류 메시지 목록 (빈 리스트면 검증 통과)
        """
        errors = []
        
        # 1. 모델 검증 (카탈로그가 있는 경우)
        if self.catalog:
            task = recipe.get_task_type().capitalize()
            model_class = recipe.model.class_path.split('.')[-1]
            
            spec = self.catalog.get_model_spec(task, model_class)
            if not spec:
                available = self.catalog.list_models_for_task(task)
                if available:  # 해당 태스크에 모델이 있는 경우만
                    errors.append(
                        f"모델 '{model_class}'이 '{task}' 카탈로그에 없습니다. "
                        f"사용 가능: {available}"
                    )
            elif not spec.is_compatible_with_task(recipe.get_task_type()):
                errors.append(
                    f"모델 '{model_class}'은 '{recipe.get_task_type()}' 태스크와 호환되지 않습니다"
                )
        
        # 2. 메트릭 검증
        task_type = recipe.get_task_type()
        valid_metrics = self.VALID_METRICS.get(task_type, [])
        
        for metric in recipe.evaluation.metrics:
            # 기본 메트릭 이름 추출 (weighted 변형 처리)
            base_metric = metric.split("_")[0] if "_" in metric else metric
            
            if metric not in valid_metrics and base_metric not in valid_metrics:
                errors.append(
                    f"메트릭 '{metric}'은 '{task_type}' 태스크에서 지원되지 않습니다. "
                    f"사용 가능: {valid_metrics[:5]}..."  # 처음 5개만 표시
                )
        
        # 3. 하이퍼파라미터 튜닝 검증
        if recipe.is_tuning_enabled():
            hp = recipe.model.hyperparameters
            
            if not hp.tunable:
                errors.append("튜닝이 활성화되었지만 tunable 파라미터가 없습니다")
            else:
                # tunable 파라미터 구조 검증
                for param, spec in hp.tunable.items():
                    if 'type' not in spec:
                        errors.append(f"Tunable 파라미터 '{param}'에 'type'이 없습니다")
                    if 'range' not in spec:
                        errors.append(f"Tunable 파라미터 '{param}'에 'range'가 없습니다")
        
        # 4. 전처리 검증
        if recipe.preprocessor:
            for i, step in enumerate(recipe.preprocessor.steps):
                # SimpleImputer는 strategy 필수
                if step.type == 'simple_imputer' and not step.strategy:
                    errors.append(f"전처리 단계 {i+1}: simple_imputer는 strategy가 필요합니다")
                
                # 컬럼 검증
                if not step.columns:
                    errors.append(f"전처리 단계 {i+1}: columns가 비어있습니다")
        
        # 5. Fetcher 검증
        if recipe.data.fetcher.type == "feature_store":
            if not recipe.data.fetcher.feature_views:
                logger.warning("feature_store fetcher에 feature_views가 정의되지 않았습니다")
        
        # 6. 모델 클래스 임포트 검증
        self._validate_model_import(recipe.model.class_path, errors)
        
        return errors
    
    def validate_config(self, config) -> List[str]:
        """
        Config 검증
        
        Args:
            config: 검증할 Config 객체
            
        Returns:
            오류 메시지 목록 (빈 리스트면 검증 통과)
        """
        errors = []
        
        # 1. 환경 이름 검증
        if not config.environment.name:
            errors.append("환경 이름이 비어있습니다")
        
        # 2. MLflow 설정 검증 (선택사항)
        if config.mlflow:
            if not config.mlflow.tracking_uri:
                errors.append("MLflow tracking_uri가 비어있습니다")
            if not config.mlflow.experiment_name:
                errors.append("MLflow experiment_name이 비어있습니다")
        
        # 3. 데이터 소스 검증
        if config.data_source.adapter_type == "sql":
            if 'connection_uri' not in config.data_source.config:
                errors.append("SQL adapter는 connection_uri가 필요합니다")
        elif config.data_source.adapter_type == "bigquery":
            required = ['project_id', 'dataset_id']
            missing = [f for f in required if f not in config.data_source.config]
            if missing:
                errors.append(f"BigQuery adapter 필수 필드: {missing}")
        
        # 4. Feature Store 검증
        if config.feature_store.provider == "feast":
            if not config.feature_store.feast_config:
                errors.append("Feast provider는 feast_config가 필요합니다")
            else:
                fc = config.feature_store.feast_config
                if not fc.project:
                    errors.append("Feast project 이름이 비어있습니다")
                if not fc.registry:
                    errors.append("Feast registry 경로가 비어있습니다")
        
        # 5. 서빙 설정 검증
        if config.serving and config.serving.enabled:
            if config.serving.port < 1024 or config.serving.port > 65535:
                errors.append(f"포트 {config.serving.port}는 유효하지 않습니다 (1024-65535)")
            if config.serving.workers < 1:
                errors.append(f"워커 수는 1 이상이어야 합니다")
        
        # 6. Artifact Store 검증
        if config.artifact_store:
            if config.artifact_store.type == "local":
                if 'base_path' not in config.artifact_store.config:
                    errors.append("Local artifact store는 base_path가 필요합니다")
            elif config.artifact_store.type in ["s3", "gcs"]:
                if 'bucket' not in config.artifact_store.config:
                    errors.append(f"{config.artifact_store.type} artifact store는 bucket이 필요합니다")
        
        return errors
    
    def validate_settings(self, settings) -> List[str]:
        """
        Settings 전체 검증 (Config + Recipe + 상호 호환성)
        
        Args:
            settings: 검증할 Settings 객체
            
        Returns:
            오류 메시지 목록 (빈 리스트면 검증 통과)
        """
        errors = []
        
        # Config 검증
        config_errors = self.validate_config(settings.config)
        if config_errors:
            errors.extend([f"[Config] {e}" for e in config_errors])
        
        # Recipe 검증
        recipe_errors = self.validate_recipe(settings.recipe)
        if recipe_errors:
            errors.extend([f"[Recipe] {e}" for e in recipe_errors])
        
        # 상호 호환성 검증
        compatibility_errors = self._validate_compatibility(settings)
        if compatibility_errors:
            errors.extend([f"[Compatibility] {e}" for e in compatibility_errors])
        
        return errors
    
    def validate(self, settings) -> None:
        """
        Settings 검증 (기존 API 호환성)
        
        Args:
            settings: Settings 객체
            
        Raises:
            ValueError: 검증 실패시
        """
        errors = self.validate_settings(settings)
        if errors:
            error_msg = "Settings 검증 실패:\n" + "\n".join(errors)
            raise ValueError(error_msg)
    
    def _validate_compatibility(self, settings) -> List[str]:
        """Config와 Recipe 간 호환성 검증"""
        errors = []
        
        # Feature Store 호환성
        if settings.recipe.data.fetcher.type == "feature_store":
            if settings.config.feature_store.provider != "feast":
                errors.append(
                    "Recipe가 feature_store를 사용하지만 Config에서 Feast가 활성화되지 않았습니다"
                )
        
        # 데이터 어댑터 호환성
        loader_adapter = settings.recipe.data.loader.get_adapter_type()
        config_adapter = settings.config.data_source.adapter_type
        
        # SQL은 모든 SQL 타입과 호환
        if loader_adapter == "sql" and config_adapter not in ["sql", "bigquery"]:
            errors.append(
                f"Recipe loader가 SQL 타입이지만 Config adapter가 {config_adapter}입니다"
            )
        elif loader_adapter == "storage" and config_adapter != "storage":
            errors.append(
                f"Recipe loader가 storage 타입이지만 Config adapter가 {config_adapter}입니다"
            )
        
        return errors
    
    def _validate_model_import(self, class_path: str, errors: List[str]) -> None:
        """모델 클래스 임포트 가능성 검증"""
        # 테스트/더미 모델은 건너뜀
        if class_path in ["inference.model", "dummy.model"] or class_path.startswith("test."):
            return
        
        try:
            # 모듈과 클래스 분리
            parts = class_path.rsplit(".", 1)
            if len(parts) != 2:
                errors.append(f"잘못된 클래스 경로 형식: {class_path}")
                return
            
            module_name, class_name = parts
            
            # 모듈 임포트 시도
            module = importlib.import_module(module_name)
            
            # 클래스 존재 확인
            if not hasattr(module, class_name):
                errors.append(f"모듈 '{module_name}'에 클래스 '{class_name}'이 없습니다")
                
        except ImportError as e:
            # 임포트 실패는 경고만 (라이브러리가 설치되지 않았을 수 있음)
            logger.warning(f"모델 클래스 임포트 실패 '{class_path}': {e}")


# 하위 호환성 유지용 별칭
validate = Validator().validate  # Static method처럼 사용 가능