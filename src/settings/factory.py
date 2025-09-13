"""통합 Settings Factory - 모든 CLI 명령어의 Settings 생성 중앙화"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import Config
from .recipe import Recipe
from .validation import ValidationOrchestrator
from .mlflow_restore import MLflowRecipeRestorer
from src.utils.core.logger import logger


class Settings:
    """통합 Settings 컨테이너"""
    
    def __init__(self, config: Config, recipe: Recipe):
        """
        Settings 초기화
        
        Args:
            config: 인프라 설정
            recipe: 워크플로우 정의
        """
        self.config = config
        self.recipe = recipe
        # 검증은 Factory에서 사전 실행됨


class SettingsFactory:
    """통합 Settings Factory - CLI 명령어별 Settings 생성"""

    def __init__(self):
        """검증 시스템 초기화"""
        self.validator = ValidationOrchestrator()

    @classmethod
    def for_training(cls, recipe_path: str, config_path: str,
                    data_path: str = None, context_params: Optional[Dict] = None) -> Settings:
        """
        train 명령어용 Settings 생성

        통합 기능:
        1. Recipe/Config 파일 로딩
        2. data_path 처리 (Jinja 템플릿 렌더링)
        3. 동적 검증 실행 (Catalog + Registry + Compatibility)
        4. 계산 필드 추가 (run_name 등)
        """
        factory = cls()
        logger.info(f"학습용 Settings 생성: recipe={recipe_path}, config={config_path}")

        # 1. 파일 로딩
        config = factory._load_config(config_path)
        recipe = factory._load_recipe(recipe_path)

        # 2. 학습 전용 데이터 경로 처리
        if data_path:
            factory._process_training_data_path(recipe, data_path, context_params)

        # 3. 동적 검증 실행
        validation_result = factory.validator.validate_for_training(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"학습 설정 검증 실패: {validation_result.error_message}")

        # 경고가 있으면 로그에 출력
        for warning in validation_result.warnings:
            logger.warning(warning)

        # 4. Settings 생성 및 계산 필드 추가
        settings = Settings(config, recipe)
        factory._add_training_computed_fields(settings, recipe_path, context_params)

        logger.info(f"학습용 Settings 생성 완료: {settings.recipe.name}")
        return settings

    @classmethod
    def for_serving(cls, config_path: str, run_id: str) -> Settings:
        """
        serve-api 명령어용 Settings 생성

        핵심 기능:
        1. 현재 Config 로딩 (서빙 환경)
        2. MLflow에서 학습시 Recipe 완전 복원
        3. 서빙 호환성 검증
        """
        factory = cls()
        logger.info(f"서빙용 Settings 생성: config={config_path}, run_id={run_id}")

        # 1. 현재 서빙 환경의 Config 로딩
        config = factory._load_config(config_path)

        # 2. MLflow Recipe 복원
        recipe_restorer = MLflowRecipeRestorer(run_id)
        recipe = recipe_restorer.restore_recipe()

        # 3. 서빙 호환성 검증
        validation_result = factory.validator.validate_for_serving(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"서빙 설정 검증 실패: {validation_result.error_message}")

        # 4. Settings 생성
        settings = Settings(config, recipe)
        factory._add_serving_computed_fields(settings, run_id)

        logger.info(f"서빙용 Settings 생성 완료")
        return settings

    @classmethod
    def for_inference(cls, config_path: str, run_id: str, data_path: str = None,
                     context_params: Optional[Dict] = None) -> Settings:
        """
        batch-inference 명령어용 Settings 생성

        핵심 기능:
        1. 현재 Config 로딩 (추론 환경)
        2. MLflow에서 학습시 Recipe 완전 복원
        3. 추론 데이터 경로 처리 (배치별 데이터)
        4. 추론 호환성 검증
        """
        factory = cls()
        logger.info(f"추론용 Settings 생성: config={config_path}, run_id={run_id}")

        # 1. 현재 추론 환경의 Config 로딩
        config = factory._load_config(config_path)

        # 2. MLflow Recipe 복원
        recipe_restorer = MLflowRecipeRestorer(run_id)
        recipe = recipe_restorer.restore_recipe()

        # 3. 추론 전용 데이터 경로 처리
        if data_path:
            factory._process_inference_data_path(recipe, data_path, context_params)

        # 4. 추론 호환성 검증
        validation_result = factory.validator.validate_for_inference(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"추론 설정 검증 실패: {validation_result.error_message}")

        # 5. Settings 생성
        settings = Settings(config, recipe)
        factory._add_inference_computed_fields(settings, run_id, data_path)

        logger.info(f"추론용 Settings 생성 완료")
        return settings

    # === 내부 유틸리티 메서드들 ===
    def _load_config(self, config_path: str) -> Config:
        """Config 파일 로딩 및 환경변수 치환"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            # 대체 경로 시도 (configs/base.yaml)
            base_path = Path("configs") / "base.yaml"
            if base_path.exists():
                logger.warning(f"Config 파일을 찾을 수 없어 base.yaml을 사용합니다: {config_path}")
                config_path = base_path
            else:
                raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
        
        # YAML 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Config 파일이 비어있습니다: {config_path}")
        
        # 환경변수 치환
        config_data = self._resolve_env_variables(config_data)
        
        # Config 객체 생성
        try:
            config = Config(**config_data)
            logger.debug(f"Config 로드 성공: {config_path}")
            return config
        except Exception as e:
            raise ValueError(f"Config 파싱 실패 ({config_path}): {str(e)}")

    def _load_recipe(self, recipe_path: str) -> Recipe:
        """Recipe 파일 로딩 및 환경변수 치환"""
        recipe_path = Path(recipe_path)
        
        # 확장자 추가 (.yaml 또는 .yml)
        if not recipe_path.suffix:
            if Path(f"{recipe_path}.yaml").exists():
                recipe_path = Path(f"{recipe_path}.yaml")
            elif Path(f"{recipe_path}.yml").exists():
                recipe_path = Path(f"{recipe_path}.yml")
            else:
                recipe_path = recipe_path.with_suffix(".yaml")
        
        # 상대 경로인 경우 recipes/ 디렉토리에서 찾기
        if not recipe_path.exists() and not recipe_path.is_absolute():
            recipes_path = Path("recipes") / recipe_path.name
            if recipes_path.exists():
                recipe_path = recipes_path
            else:
                raise FileNotFoundError(f"Recipe 파일을 찾을 수 없습니다: {recipe_path}")
        
        # YAML 로드
        with open(recipe_path, 'r', encoding='utf-8') as f:
            recipe_data = yaml.safe_load(f)
        
        if not recipe_data:
            raise ValueError(f"Recipe 파일이 비어있습니다: {recipe_path}")
        
        # 환경변수 치환
        recipe_data = self._resolve_env_variables(recipe_data)
        
        # Recipe 객체 생성
        try:
            recipe = Recipe(**recipe_data)
            logger.debug(f"Recipe 로드 성공: {recipe_path}")
            return recipe
        except Exception as e:
            raise ValueError(f"Recipe 파싱 실패 ({recipe_path}): {str(e)}")

    def _resolve_env_variables(self, data: Any) -> Any:
        """환경변수 치환 - ${VAR:default} 패턴 지원"""
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
            return {k: self._resolve_env_variables(v) for k, v in data.items()}
        
        elif isinstance(data, list):
            # 리스트도 재귀적으로 처리
            return [self._resolve_env_variables(item) for item in data]
        
        # 다른 타입은 그대로 반환
        return data

    def _process_training_data_path(self, recipe: Recipe, data_path: str,
                                   context_params: Optional[Dict]) -> None:
        """학습용 데이터 경로 처리 (Jinja 템플릿 렌더링)"""
        if not data_path:
            logger.debug("data_path가 없어서 건너뜁니다")
            return

        # Jinja 템플릿 처리
        if data_path.endswith('.sql.j2') or (data_path.endswith('.sql') and context_params):
            data_path = self._render_jinja_template(data_path, context_params)

        # data_path를 recipe.data.loader.source_uri에 주입
        recipe.data.loader.source_uri = data_path
        logger.debug(f"학습 데이터 경로 설정: {data_path}")

    def _process_inference_data_path(self, recipe: Recipe, data_path: str,
                                    context_params: Optional[Dict]) -> None:
        """추론용 데이터 경로 처리 (Jinja 템플릿 렌더링)"""
        if not data_path:
            logger.debug("data_path가 없어서 건너뜁니다")
            return

        # Jinja 템플릿 처리
        if data_path.endswith('.sql.j2') or (data_path.endswith('.sql') and context_params):
            data_path = self._render_jinja_template(data_path, context_params)

        # data_path를 recipe.data.loader.source_uri에 주입
        recipe.data.loader.source_uri = data_path
        logger.debug(f"추론 데이터 경로 설정: {data_path}")

    def _add_training_computed_fields(self, settings: Settings, recipe_path: str,
                                     context_params: Optional[Dict]) -> None:
        """학습용 계산 필드 추가"""
        # run_name 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recipe_name = Path(recipe_path).stem
        run_name = f"{recipe_name}_{timestamp}"
        
        # Recipe에 computed 필드가 없으면 생성
        if not hasattr(settings.recipe.model, 'computed') or not settings.recipe.model.computed:
            settings.recipe.model.computed = {}
        
        settings.recipe.model.computed.update({
            "run_name": run_name,
            "environment": settings.config.environment.name,
            "recipe_file": recipe_path
        })

    def _add_serving_computed_fields(self, settings: Settings, run_id: str) -> None:
        """서빙용 계산 필드 추가"""
        if not hasattr(settings.recipe.model, 'computed') or not settings.recipe.model.computed:
            settings.recipe.model.computed = {}
            
        settings.recipe.model.computed.update({
            "run_id": run_id,
            "environment": settings.config.environment.name,
            "mode": "serving"
        })

    def _add_inference_computed_fields(self, settings: Settings, run_id: str, data_path: str) -> None:
        """추론용 계산 필드 추가"""
        if not hasattr(settings.recipe.model, 'computed') or not settings.recipe.model.computed:
            settings.recipe.model.computed = {}
            
        settings.recipe.model.computed.update({
            "run_id": run_id,
            "environment": settings.config.environment.name,
            "mode": "inference",
            "data_path": data_path
        })

    def _create_minimal_recipe_for_serving(self) -> Recipe:
        """서빙용 최소 Recipe 생성 (MLflow 복원 전까지 임시)"""
        from .recipe import (
            Recipe, Model, Data, Loader, Fetcher, DataInterface, DataSplit,
            Evaluation, Metadata, HyperparametersTuning
        )
        
        return Recipe(
            name="serving_recipe",
            task_choice="classification",
            model=Model(
                class_path="sklearn.ensemble.RandomForestClassifier",
                library="sklearn",
                hyperparameters=HyperparametersTuning(tuning_enabled=False, values={"n_estimators": 100})
            ),
            data=Data(
                loader=Loader(source_uri=None),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(target_column="target", entity_columns=["id"]),
                split=DataSplit(train=0.8, test=0.1, validation=0.1)
            ),
            evaluation=Evaluation(metrics=["accuracy"], random_state=42),
            metadata=Metadata(
                author="SettingsFactory",
                created_at=datetime.now().isoformat(),
                description="Serving용 최소 Recipe"
            )
        )

    def _create_minimal_recipe_for_inference(self) -> Recipe:
        """추론용 최소 Recipe 생성 (MLflow 복원 전까지 임시)"""
        return self._create_minimal_recipe_for_serving()  # 동일한 구조 사용

    def _render_jinja_template(self, data_path: str, context_params: Optional[Dict]) -> str:
        """Jinja 템플릿 렌더링"""
        from src.utils.template.templating_utils import render_template_from_string

        template_path = Path(data_path)
        if not template_path.exists():
            raise FileNotFoundError(f"템플릿 파일을 찾을 수 없습니다: {data_path}")

        template_content = template_path.read_text()

        if not context_params:
            if data_path.endswith('.sql.j2'):
                raise ValueError(f"Jinja 템플릿 파일({data_path})에는 context_params가 필요합니다")
            # .sql 파일에 params 없으면 그대로 반환
            return template_content

        try:
            rendered_content = render_template_from_string(template_content, context_params)
            logger.info(f"✅ Jinja 템플릿 렌더링 성공: {data_path}")
            return rendered_content
        except ValueError as e:
            logger.error(f"🚨 Jinja 렌더링 실패: {e}")
            raise ValueError(f"템플릿 렌더링 실패: {e}")


# 하위 호환성 편의 함수
def load_settings(recipe_path: str, config_path: str, **kwargs) -> Settings:
    """하위 호환성: 기존 load_settings() 지원"""
    return SettingsFactory.for_training(
        recipe_path=recipe_path, 
        config_path=config_path,
        data_path=kwargs.get('data_path')
    )
