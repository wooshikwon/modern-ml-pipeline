"""
Settings Loader - CLI 호환 로딩 로직
YAML 구조를 정확히 파싱하고 검증
"""

import os
import re
from pathlib import Path
from typing import Dict, Any
import yaml
from datetime import datetime

from .config import Config
from .recipe import Recipe
from src.utils.core.logger import logger


class Settings:
    """
    통합 Settings 컨테이너
    Config(인프라)와 Recipe(워크플로우)를 함께 관리
    """
    
    def __init__(self, config: Config, recipe: Recipe):
        """
        Settings 초기화 및 검증
        
        Args:
            config: 인프라 설정
            recipe: 워크플로우 정의
        """
        self.config = config
        self.recipe = recipe
        self._validate()
    
    def _validate(self) -> None:
        """Config와 Recipe 간 호환성 검증"""
        
        # 1. Feature Store 일관성 체크
        if self.recipe.data.fetcher.type == "feature_store":
            if self.config.feature_store.provider != "feast":
                raise ValueError(
                    "Recipe에서 feature_store fetcher를 사용하지만 "
                    f"Config의 feature_store provider가 '{self.config.feature_store.provider}'입니다. "
                    "'feast'로 설정해주세요."
                )
            
            # Feast config 존재 확인
            if not self.config.feature_store.feast_config:
                raise ValueError(
                    "feature_store fetcher 사용시 Config에 feast_config가 필요합니다"
                )
        
        # 2. 데이터 소스 타입 호환성 체크는 source_uri 주입 후 별도 검증
        
        # 3. MLflow 설정 체크 (선택사항이지만 권장)
        if not self.config.mlflow:
            logger.warning("MLflow가 설정되지 않았습니다. 실험 추적이 비활성화됩니다.")
    
    def validate_data_source_compatibility(self) -> None:
        """
        CLI에서 source_uri 주입 후 config.data_source와 호환성 검증.
        
        Raises:
            ValueError: Config adapter type과 source_uri가 호환되지 않는 경우
        """
        if not hasattr(self.recipe.data.loader, 'source_uri') or not self.recipe.data.loader.source_uri:
            # source_uri가 아직 주입되지 않았으면 검증 스킵
            return
            
        source_uri = self.recipe.data.loader.source_uri.lower()
        config_adapter = self.config.data_source.adapter_type
        
        # 파일 확장자/패턴 기반 adapter type 추론
        if source_uri.endswith('.sql') or 'select' in source_uri or 'from' in source_uri:
            # SQL 쿼리/파일
            detected_type = 'sql'
            compatible_types = ['sql', 'bigquery']
            
            if config_adapter not in compatible_types:
                raise ValueError(
                    f"SQL 데이터 소스({self.recipe.data.loader.source_uri})는 "
                    f"Config adapter_type이 {compatible_types} 중 하나여야 합니다. "
                    f"현재: {config_adapter}"
                )
                
        elif (source_uri.endswith('.csv') or source_uri.endswith('.parquet') or 
              source_uri.endswith('.json') or source_uri.startswith('s3://') or
              source_uri.startswith('gs://') or source_uri.startswith('az://')):
            # Storage 기반 파일
            detected_type = 'storage'
            
            if config_adapter != 'storage':
                raise ValueError(
                    f"Storage 파일({self.recipe.data.loader.source_uri})은 "
                    f"Config adapter_type이 'storage'여야 합니다. "
                    f"현재: {config_adapter}"
                )
                
        elif source_uri.startswith('bigquery://'):
            # BigQuery 직접 연결
            detected_type = 'bigquery'
            
            if config_adapter != 'bigquery':
                raise ValueError(
                    f"BigQuery URI({self.recipe.data.loader.source_uri})는 "
                    f"Config adapter_type이 'bigquery'여야 합니다. "
                    f"현재: {config_adapter}"
                )
        
        from src.utils.core.logger import logger
        logger.info(f"✅ 데이터 소스 호환성 검증 완료: {detected_type} ↔ {config_adapter}")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화용)"""
        return {
            "config": self.config.model_dump(),
            "recipe": self.recipe.model_dump()
        }
    
    def get_environment_name(self) -> str:
        """환경 이름 반환"""
        return self.config.environment.name
    
    def get_recipe_name(self) -> str:
        """레시피 이름 반환"""
        return self.recipe.name


def resolve_env_variables(data: Any) -> Any:
    """
    환경변수 치환 - ${VAR:default} 패턴 지원
    CLI 템플릿에서 사용하는 패턴과 호환
    
    Args:
        data: 환경변수를 치환할 데이터 (문자열, 딕셔너리, 리스트 등)
        
    Returns:
        환경변수가 치환된 데이터
    
    Examples:
        "${DB_HOST:localhost}" -> "localhost" (환경변수 없을 때)
        "${DB_PORT:5432}" -> 5432 (정수로 변환)
        "${DEBUG:false}" -> False (불린으로 변환)
    """
    if isinstance(data, str):
        # ${VAR:default} 패턴 매칭
        pattern = r'\$\{([^}]+)\}'
        
        # 전체 문자열이 환경변수인지 확인
        full_match = re.fullmatch(pattern, data)
        
        if full_match:
            # 전체가 환경변수인 경우 - 타입 변환 시도
            expr = full_match.group(1)
            
            # 콜론으로 변수명과 기본값 분리
            if ':' in expr:
                var_name, default_value = expr.split(':', 1)
                var_name = var_name.strip()
                default_value = default_value.strip()
                result = os.environ.get(var_name, default_value)
            else:
                # 기본값 없음
                var_name = expr.strip()
                result = os.environ.get(var_name, data)  # 없으면 원본 반환
            
            # 타입 변환 시도
            if isinstance(result, str):
                # 빈 문자열 처리
                if result == "":
                    return ""
                
                # Boolean 변환
                if result.lower() in ('true', 'false'):
                    return result.lower() == 'true'
                
                # 숫자 변환
                try:
                    # 정수 변환 시도
                    if '.' not in result and 'e' not in result.lower():
                        return int(result)
                    # 실수 변환 시도
                    return float(result)
                except (ValueError, AttributeError):
                    # 변환 실패시 문자열 그대로 반환
                    return result
            
            return result
            
        else:
            # 부분적으로 환경변수가 포함된 경우 - 문자열로만 치환
            def replacer(match):
                expr = match.group(1)
                
                if ':' in expr:
                    var_name, default_value = expr.split(':', 1)
                    var_name = var_name.strip()
                    default_value = default_value.strip()
                    return str(os.environ.get(var_name, default_value))
                else:
                    var_name = expr.strip()
                    return str(os.environ.get(var_name, match.group(0)))
            
            return re.sub(pattern, replacer, data)
    
    elif isinstance(data, dict):
        # 딕셔너리는 재귀적으로 처리
        return {k: resolve_env_variables(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        # 리스트도 재귀적으로 처리
        return [resolve_env_variables(item) for item in data]
    
    # 다른 타입은 그대로 반환
    return data


def load_settings(recipe_path: str, config_path: str, **kwargs) -> Settings:
    """
    Settings 로드 (직접 파일 경로 방식)
    
    순서:
    1. config_path에서 직접 Config 로드
    2. recipe_path에서 직접 Recipe 로드  
    3. Settings 객체 생성 및 검증
    
    Args:
        recipe_path: Recipe YAML 파일 경로
        config_path: Config YAML 파일 경로
        **kwargs: 추가 파라미터 (호환성용, 무시됨)
        
    Returns:
        Settings 객체 (Config + Recipe)
        
    Raises:
        FileNotFoundError: 설정 파일이 없을 때
        ValueError: 검증 실패시
    """
    logger.info(f"Settings 로드 시작: recipe={recipe_path}, config={config_path}")
    
    # 1. Config 로드 (직접 경로)
    config = _load_config(config_path)
    
    # 2. Recipe 로드 (직접 경로)
    recipe = _load_recipe(recipe_path)
    
    # 4. Settings 생성 (검증 포함)
    settings = Settings(config, recipe)
    
    # 3. 런타임 필드 추가
    _add_computed_fields(settings, recipe_path)
    
    logger.info(
        f"Settings 로드 완료: "
        f"recipe={settings.recipe.name}, "
        f"env={settings.config.environment.name}, "
        f"task={settings.recipe.get_task_type()}"
    )
    
    return settings


def _load_config(config_path: str) -> Config:
    """
    Config 파일 로드 및 파싱
    
    Args:
        config_path: Config 파일 경로
        
    Returns:
        Config 객체
        
    Raises:
        FileNotFoundError: config 파일이 없을 때
    """
    # 직접 경로 사용
    config_path = Path(config_path)
    
    if not config_path.exists():
        # 대체 경로 시도 (configs/base.yaml)
        base_path = Path("configs") / "base.yaml"
        if base_path.exists():
            logger.warning(
                "Config file not found, using base.yaml instead. "
                "Create proper config with 'mmp get-config --env-name <env_name>'."
            )
            config_path = base_path
        else:
            raise FileNotFoundError(
                f"Config 파일을 찾을 수 없습니다: {config_path}\n"
                f"다음 명령으로 생성하세요:\n"
                f"  mmp get-config --env-name <env_name>"
            )
    
    # YAML 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    if not config_data:
        raise ValueError(f"Config 파일이 비어있습니다: {config_path}")
    
    # 환경변수 치환
    config_data = resolve_env_variables(config_data)
    
    # Config 객체 생성
    try:
        config = Config(**config_data)
        logger.info(f"Config 로드 성공: {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Config 파싱 실패 ({config_path}): {str(e)}")


def _load_recipe(recipe_file: str) -> Recipe:
    """
    Recipe 파일 로드 및 파싱
    
    Args:
        recipe_file: Recipe 파일 경로
        
    Returns:
        Recipe 객체
        
    Raises:
        FileNotFoundError: recipe 파일이 없을 때
    """
    recipe_path = Path(recipe_file)
    
    # 확장자 추가 (.yaml 또는 .yml)
    if not recipe_path.suffix:
        if Path(f"{recipe_file}.yaml").exists():
            recipe_path = Path(f"{recipe_file}.yaml")
        elif Path(f"{recipe_file}.yml").exists():
            recipe_path = Path(f"{recipe_file}.yml")
        else:
            recipe_path = recipe_path.with_suffix(".yaml")
    
    # 상대 경로인 경우 recipes/ 디렉토리에서 찾기
    if not recipe_path.exists() and not recipe_path.is_absolute():
        recipes_path = Path("recipes") / recipe_path.name
        if recipes_path.exists():
            recipe_path = recipes_path
        else:
            raise FileNotFoundError(
                f"Recipe 파일을 찾을 수 없습니다: {recipe_file}\n"
                f"시도한 경로:\n"
                f"  - {recipe_path}\n"
                f"  - {recipes_path}\n"
                f"'mmp get-recipe'로 새 recipe를 생성하세요."
            )
    
    # YAML 로드
    with open(recipe_path, 'r', encoding='utf-8') as f:
        recipe_data = yaml.safe_load(f)
    
    if not recipe_data:
        raise ValueError(f"Recipe 파일이 비어있습니다: {recipe_path}")
    
    # 환경변수 치환 (Recipe에서는 보통 없지만 지원)
    recipe_data = resolve_env_variables(recipe_data)
    
    # Recipe 객체 생성
    try:
        recipe = Recipe(**recipe_data)
        logger.info(f"Recipe 로드 성공: {recipe_path}")
        return recipe
    except Exception as e:
        raise ValueError(f"Recipe 파싱 실패 ({recipe_path}): {str(e)}")


def _add_computed_fields(settings: Settings, recipe_file: str) -> None:
    """
    런타임 계산 필드 추가
    
    Args:
        settings: Settings 객체
        recipe_file: 원본 recipe 파일 경로 (run_name 생성용)
    """
    # computed 필드 초기화
    if not settings.recipe.model.computed:
        settings.recipe.model.computed = {}
    
    # run_name 생성 (없는 경우)
    if "run_name" not in settings.recipe.model.computed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recipe_name = Path(recipe_file).stem
        run_name = f"{recipe_name}_{timestamp}"
        settings.recipe.model.computed["run_name"] = run_name
        logger.debug(f"Run name 생성: {run_name}")
    
    # 환경 정보 추가
    settings.recipe.model.computed["environment"] = settings.config.environment.name
    
    # 튜닝 정보 추가
    if settings.recipe.is_tuning_enabled():
        settings.recipe.model.computed["tuning_enabled"] = True
        tunable = settings.recipe.get_tunable_params()
        if tunable:
            settings.recipe.model.computed["tunable_params"] = list(tunable.keys())


def create_settings_for_inference(config_data: Dict[str, Any]) -> Settings:
    """
    추론/서빙용 최소 Settings 생성
    Recipe 없이 Config만으로 Settings 생성
    
    Args:
        config_data: Config 딕셔너리 데이터
        
    Returns:
        최소 Settings 객체
    """
    # Config 생성
    config = Config(**config_data)
    
    # 최소 Recipe 생성
    recipe = Recipe(
        name="inference",
        model={
            "class_path": "inference.model",
            "library": "sklearn",
            "hyperparameters": {
                "tuning_enabled": False,
                "values": {}
            }
        },
        data={
            "loader": {
                "source_uri": "inference_data",
                "entity_schema": {
                    "entity_columns": ["id"],
                    "timestamp_column": "timestamp"
                }
            },
            "fetcher": {
                "type": "pass_through",
                "features": None
            },
            "data_interface": {
                "task_type": "classification",
                "target_column": "target",
                "entity_columns": ["id"]
            }
        },
        evaluation={
            "metrics": ["accuracy"],
            "validation": {
                "method": "train_test_split",
                "test_size": 0.2
            }
        }
    )
    
    return Settings(config, recipe)


def load_config_files(config_path: str) -> Dict[str, Any]:
    """
    Config 파일만 로드 (서빙/추론용)
    Recipe 없이 Config만 필요한 경우 사용
    
    Args:
        config_path: Config 파일 경로
        
    Returns:
        Config 딕셔너리
    """
    # Config 로드 (직접 경로)
    config = _load_config(config_path)
    
    # 딕셔너리로 변환
    return config.model_dump()


# load_settings_by_file 별칭 제거됨 - 직접 load_settings 사용