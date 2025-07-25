import yaml
import importlib
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import mlflow
import pandas as pd

from src.core.augmenter import Augmenter, LocalFileAugmenter, BaseAugmenter, PassThroughAugmenter
from src.core.preprocessor import BasePreprocessor, Preprocessor
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger
from src.utils.adapters.file_system_adapter import FileSystemAdapter
from src.utils.adapters.bigquery_adapter import BigQueryAdapter
from src.utils.adapters.gcs_adapter import GCSAdapter
from src.utils.adapters.s3_adapter import S3Adapter
# 🆕 Blueprint v17.0: Registry 패턴 import
from src.core.registry import AdapterRegistry

# Redis는 선택적 의존성으로 처리
try:
    from src.utils.adapters.redis_adapter import RedisAdapter
    HAS_REDIS = True
except ImportError:
    RedisAdapter = None
    HAS_REDIS = False

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    완전한 Wrapped Artifact 구현: Blueprint v17.0
    학습 시점의 모든 로직과 메타데이터를 완전히 캡슐화한 자기 완결적 아티팩트
    + 하이퍼파라미터 최적화 결과 및 Data Leakage 방지 메타데이터 포함
    """
    def __init__(
        self,
        trained_model,
        trained_preprocessor: Optional[BasePreprocessor],
        trained_augmenter: BaseAugmenter,
        loader_sql_snapshot: str,
        augmenter_sql_snapshot: str,
        recipe_yaml_snapshot: str,
        training_metadata: Dict[str, Any],
        # 🆕 새로운 인자들 (Optional로 하위 호환성 보장)
        model_class_path: Optional[str] = None,
        hyperparameter_optimization: Optional[Dict[str, Any]] = None,
        training_methodology: Optional[Dict[str, Any]] = None,
    ):
        # 학습된 컴포넌트들
        self.trained_model = trained_model
        self.trained_preprocessor = trained_preprocessor
        self.trained_augmenter = trained_augmenter
        
        # 로직의 완전한 스냅샷
        self.loader_sql_snapshot = loader_sql_snapshot
        self.augmenter_sql_snapshot = augmenter_sql_snapshot
        self.recipe_yaml_snapshot = recipe_yaml_snapshot
        
        # 메타데이터
        self.training_metadata = training_metadata
        
        # 🆕 새로운 메타데이터 (Blueprint v17.0)
        self.model_class_path = model_class_path
        self.hyperparameter_optimization = hyperparameter_optimization or {"enabled": False}
        self.training_methodology = training_methodology or {}
        
        # 하위 호환성을 위한 별칭
        self.augmenter = trained_augmenter
        self.preprocessor = trained_preprocessor
        self.model = trained_model
        self.loader_uri = training_metadata.get("loader_uri", "")
        self.recipe_snapshot = training_metadata.get("recipe_snapshot", {})

    def predict(
        self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        컨텍스트에 따른 예측 실행 (Blueprint v13.0)
        배치 추론과 API 서빙 모두 지원하는 통합 예측 엔드포인트
        """
        params = params or {}
        run_mode = params.get("run_mode", "serving")
        return_intermediate = params.get("return_intermediate", False)

        logger.info(f"PyfuncWrapper.predict 실행 시작 (모드: {run_mode})")

        # 1. 피처 증강 (Augmentation)
        if run_mode == "batch":
            # 배치 모드: SQL 직접 실행
            augmented_df = self.trained_augmenter.augment_batch(
                model_input, 
                sql_snapshot=self.augmenter_sql_snapshot,
                context_params=params.get("context_params", {})
            )
        else:
            # 실시간 모드: Feature Store 조회
            augmented_df = self.trained_augmenter.augment_realtime(
                model_input,
                sql_snapshot=self.augmenter_sql_snapshot,
                feature_store_config=params.get("feature_store_config"),
                feature_columns=params.get("feature_columns")
            )

        # 2. 전처리 (Preprocessing)
        if self.trained_preprocessor:
            preprocessed_df = self.trained_preprocessor.transform(augmented_df)
        else:
            preprocessed_df = augmented_df

        # 3. 모델 추론 (Prediction)
        predictions = self.trained_model.predict(preprocessed_df)

        # 4. 결과 정리
        results_df = model_input.merge(
            pd.DataFrame(predictions, index=model_input.index, columns=["uplift_score"]),
            left_index=True,
            right_index=True,
        )
        logger.info("PyfuncWrapper.predict 실행 완료.")

        if return_intermediate and run_mode == "batch":
            return {
                "final_results": results_df,
                "augmented_data": augmented_df,
                "preprocessed_data": preprocessed_df,
                # 🆕 최적화 메타데이터 포함 (Blueprint v17.0)
                "hyperparameter_optimization": self.hyperparameter_optimization,
                "training_methodology": self.training_methodology,
            }
        else:
            return results_df

class Factory:
    """
    설정(settings)과 URI 스킴(scheme)에 기반하여 모든 핵심 컴포넌트의 인스턴스를 생성하는 중앙 팩토리 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Factory가 초기화되었습니다.")

    def create_data_adapter(self, adapter_purpose: str = "loader", source_path: str = None) -> BaseAdapter:
        """
        🆕 Blueprint v17.0: Config-driven Dynamic Factory
        
        환경별 어댑터 매핑과 동적 생성을 통해 Blueprint 원칙을 완전히 구현합니다.
        - 원칙 1: "레시피는 논리, 설정은 인프라" - config 기반 어댑터 선택
        - 원칙 9: "환경별 차등적 기능 분리" - 동일한 논리 경로가 환경별로 다른 어댑터
        
        Args:
            adapter_purpose: 어댑터 목적 ("loader", "storage", "feature_store")
            source_path: 논리적 경로 (선택적, 어댑터 초기화에 사용)
            
        Returns:
            BaseAdapter: 환경별로 동적으로 생성된 어댑터
            
        Example:
            # LOCAL 환경: FileSystemAdapter 생성
            adapter = factory.create_data_adapter("loader", "recipes/sql/loaders/user_spine.sql")
            
            # PROD 환경: BigQueryAdapter 생성 (동일한 논리 경로)
            adapter = factory.create_data_adapter("loader", "recipes/sql/loaders/user_spine.sql")
        """
        logger.info(f"Config-driven Dynamic Factory: {adapter_purpose} 어댑터 생성 시작")
        
        # 1. data_adapters 설정 확인
        if not self.settings.data_adapters:
            raise ValueError(
                "data_adapters 설정이 없습니다. config/*.yaml에 data_adapters 섹션을 추가해주세요."
            )
        
        # 2. 목적별 기본 어댑터 조회
        try:
            adapter_name = self.settings.data_adapters.get_default_adapter(adapter_purpose)
            logger.info(f"환경 '{self.settings.environment.app_env}'에서 {adapter_purpose} 목적으로 '{adapter_name}' 어댑터 선택")
        except ValueError as e:
            raise ValueError(f"어댑터 목적 조회 실패: {e}")
        
        # 3. 어댑터 설정 조회
        try:
            adapter_config = self.settings.data_adapters.get_adapter_config(adapter_name)
            logger.info(f"어댑터 '{adapter_name}' 설정 조회 완료: {adapter_config.class_name}")
        except ValueError as e:
            raise ValueError(f"어댑터 설정 조회 실패: {e}")
        
        # 4. 동적 어댑터 클래스 import
        try:
            adapter_class = self._get_adapter_class(adapter_config.class_name)
            logger.info(f"어댑터 클래스 '{adapter_config.class_name}' 동적 import 완료")
        except Exception as e:
            raise ValueError(f"어댑터 클래스 import 실패: {adapter_config.class_name}, 오류: {e}")
        
        # 5. 어댑터 인스턴스 생성
        try:
            adapter_instance = adapter_class(
                config=adapter_config.config,
                settings=self.settings,
                source_path=source_path
            )
            logger.info(f"어댑터 인스턴스 생성 완료: {adapter_config.class_name}")
            return adapter_instance
            
        except Exception as e:
            logger.error(f"어댑터 인스턴스 생성 실패: {adapter_config.class_name}, 오류: {e}")
            # 기존 방식으로 fallback 시도
            try:
                logger.warning("기존 방식으로 fallback 시도")
                adapter_instance = adapter_class(self.settings)
                logger.info(f"Fallback 어댑터 생성 성공: {adapter_config.class_name}")
                return adapter_instance
            except Exception as fallback_error:
                raise ValueError(
                    f"어댑터 생성 실패: {adapter_config.class_name}\n"
                    f"새로운 방식 오류: {e}\n"
                    f"Fallback 오류: {fallback_error}"
                )
    
    def _get_adapter_class(self, class_name: str):
        """
        🆕 Blueprint v17.0: Registry 패턴 기반 어댑터 클래스 조회
        기존 클래스명 -> 어댑터 타입 변환 후 Registry에서 조회
        
        Args:
            class_name: 어댑터 클래스 이름 (e.g., "FileSystemAdapter")
            
        Returns:
            어댑터 클래스 객체
        """
        # 클래스명 -> 어댑터 타입 매핑 (하위 호환성)
        class_to_type_mapping = {
            "FileSystemAdapter": "filesystem",
            "BigQueryAdapter": "bigquery",
            "GCSAdapter": "gcs",
            "S3Adapter": "s3",
            "PostgreSQLAdapter": "postgresql",
            "RedisAdapter": "redis",
            "FeatureStoreAdapter": "feature_store",
            "OptunaAdapter": "optuna",
        }
        
        # 1. 클래스명을 어댑터 타입으로 변환
        adapter_type = class_to_type_mapping.get(class_name)
        if not adapter_type:
            raise ValueError(f"지원하지 않는 어댑터 클래스: {class_name}")
        
        # 2. Registry에서 어댑터 클래스 조회
        registered_adapters = AdapterRegistry.get_registered_adapters()
        if adapter_type not in registered_adapters:
            available_types = list(registered_adapters.keys())
            raise ValueError(
                f"Registry에 등록되지 않은 어댑터 타입: '{adapter_type}'\n"
                f"사용 가능한 타입: {available_types}"
            )
        
        adapter_class = registered_adapters[adapter_type]
        logger.info(f"Registry에서 어댑터 클래스 조회: {class_name} -> {adapter_type} -> {adapter_class.__name__}")
        
        return adapter_class
        
    # 🔄 기존 메서드 유지 (하위 호환성)
    def create_data_adapter_legacy(self, scheme: str) -> BaseAdapter:
        """
        🔄 기존 URI 스킴 기반 어댑터 생성 (하위 호환성 유지)
        
        ⚠️ DEPRECATED: 새로운 코드에서는 create_data_adapter() 사용 권장
        """
        logger.warning(f"DEPRECATED: URI 스킴 기반 어댑터 생성 (scheme: {scheme}). 새로운 config 기반 방식 사용 권장")
        
        if scheme == 'file':
            return FileSystemAdapter(self.settings)
        elif scheme == 'bq':
            return BigQueryAdapter(self.settings)
        elif scheme == 'gs':
            return GCSAdapter(self.settings)
        elif scheme == 's3':
            return S3Adapter(self.settings)
        else:
            raise ValueError(f"지원하지 않는 데이터 어댑터 스킴입니다: {scheme}")

    def create_redis_adapter(self):
        if not HAS_REDIS:
            logger.warning("Redis 라이브러리가 설치되지 않아 Redis 어댑터를 생성할 수 없습니다.")
            raise ImportError("Redis 라이브러리가 필요합니다. `pip install redis`로 설치하세요.")
        
        logger.info("Redis 어댑터를 생성합니다.")
        return RedisAdapter(self.settings.serving.realtime_feature_store)

    def create_augmenter(self) -> "BaseAugmenter":
        """Augmenter 생성"""
        # Blueprint 원칙 9: 환경별 차등적 기능 분리
        # LOCAL 환경에서는 PassThroughAugmenter를 강제 사용하여 빠른 실험 지원
        if self.settings.environment.app_env == "local":
            logger.info("LOCAL 환경: PassThroughAugmenter 생성 (Blueprint 원칙 9 - 의도적 제약)")
            from src.core.augmenter import PassThroughAugmenter
            return PassThroughAugmenter(settings=self.settings)
        else:
            logger.info("DEV/PROD 환경: FeatureStore 연동 Augmenter 생성")
            from src.core.augmenter import Augmenter
            return Augmenter(settings=self.settings, factory=self)

    def create_preprocessor(self) -> "BasePreprocessor":
        preprocessor_config = self.settings.model.preprocessor
        if not preprocessor_config: return None
        return Preprocessor(config=preprocessor_config, settings=self.settings)

    def create_model(self):
        """
        외부 라이브러리 직접 로딩 기반 동적 모델 생성 시스템
        무제한적인 YAML 기반 실험 자유도를 제공합니다.
        """
        class_path = self.settings.model.class_path
        
        try:
            # 모듈 경로와 클래스 이름 분리
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 동적 모듈 로드
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # 하이퍼파라미터 전달하여 인스턴스 생성
            hyperparams = self.settings.model.hyperparameters.root
            
            logger.info(f"외부 모델 로딩: {class_path}")
            return model_class(**hyperparams)
                
        except Exception as e:
            logger.error(f"모델 로딩 실패: {class_path}, 오류: {e}")
            raise ValueError(f"모델 클래스를 로드할 수 없습니다: {class_path}") from e

    def create_evaluator(self):
        """task_type에 따른 동적 evaluator 생성"""
        # Dynamic import로 순환 참조 방지
        from src.core.evaluator import (
            ClassificationEvaluator,
            RegressionEvaluator,
            ClusteringEvaluator,
            CausalEvaluator,
        )
        
        task_type = self.settings.model.data_interface.task_type
        data_interface = self.settings.model.data_interface
        
        evaluator_map = {
            "classification": ClassificationEvaluator,
            "regression": RegressionEvaluator,
            "clustering": ClusteringEvaluator,
            "causal": CausalEvaluator,
        }
        
        if task_type not in evaluator_map:
            supported_types = list(evaluator_map.keys())
            raise ValueError(
                f"지원하지 않는 task_type: '{task_type}'. "
                f"지원 가능한 타입: {supported_types}"
            )
        
        logger.info(f"'{task_type}' 타입용 evaluator를 생성합니다.")
        return evaluator_map[task_type](data_interface)

    # 🆕 새로운 메서드들 추가
    def create_feature_store_adapter(self):
        """환경별 Feature Store 어댑터 생성"""
        if not self.settings.feature_store:
            raise ValueError("Feature Store 설정이 없습니다.")
        
        logger.info("Feature Store 어댑터를 생성합니다.")
        from src.utils.adapters.feature_store_adapter import FeatureStoreAdapter
        return FeatureStoreAdapter(self.settings)
    
    def create_optuna_adapter(self):
        """Optuna SDK 래퍼 생성"""
        if not self.settings.hyperparameter_tuning:
            raise ValueError("Hyperparameter tuning 설정이 없습니다.")
        
        logger.info("Optuna 어댑터를 생성합니다.")
        from src.utils.adapters.optuna_adapter import OptunaAdapter
        return OptunaAdapter(self.settings.hyperparameter_tuning)
    
    def create_tuning_utils(self):
        """하이퍼파라미터 튜닝 유틸리티 생성"""
        logger.info("Tuning 유틸리티를 생성합니다.")
        from src.utils.system.tuning_utils import TuningUtils
        return TuningUtils()

    def create_pyfunc_wrapper(
        self, 
        trained_model, 
        trained_preprocessor: Optional[BasePreprocessor],
        training_results: Optional[Dict[str, Any]] = None  # 🆕 Trainer 결과 전달
    ) -> PyfuncWrapper:
        """
        완전한 Wrapped Artifact 생성 (Blueprint v17.0)
        학습 시점의 모든 로직과 메타데이터를 완전히 캡슐화
        + 하이퍼파라미터 최적화 결과 및 Data Leakage 방지 메타데이터 포함
        """
        logger.info("완전한 Wrapped Artifact 생성을 시작합니다.")
        
        # 1. 학습된 Augmenter 생성
        trained_augmenter = self.create_augmenter()
        
        # 2. SQL 스냅샷 생성
        loader_sql_snapshot = self._create_loader_sql_snapshot()
        augmenter_sql_snapshot = self._create_augmenter_sql_snapshot()
        
        # 3. Recipe YAML 스냅샷 생성
        recipe_yaml_snapshot = self._create_recipe_yaml_snapshot()
        
        # 4. 메타데이터 생성
        training_metadata = self._create_training_metadata()
        
        # 🆕 5. 새로운 메타데이터 처리 (Blueprint v17.0)
        model_class_path = self.settings.model.class_path
        hyperparameter_optimization = None
        training_methodology = None
        
        if training_results:
            hyperparameter_optimization = training_results.get('hyperparameter_optimization')
            training_methodology = training_results.get('training_methodology')
        
        # 6. 확장된 Wrapper 생성 (하위 호환성 유지)
        return PyfuncWrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_augmenter=trained_augmenter,
            loader_sql_snapshot=loader_sql_snapshot,
            augmenter_sql_snapshot=augmenter_sql_snapshot,
            recipe_yaml_snapshot=recipe_yaml_snapshot,
            training_metadata=training_metadata,
            # 🆕 새로운 인자들
            model_class_path=model_class_path,
            hyperparameter_optimization=hyperparameter_optimization,
            training_methodology=training_methodology,
        )
    
    def _create_loader_sql_snapshot(self) -> str:
        """Loader SQL 스냅샷 생성"""
        loader_uri = self.settings.model.loader.source_uri
        
        if loader_uri.startswith("bq://"):
            # SQL 파일 경로 추출
            sql_path = loader_uri.replace("bq://", "")
            sql_file = Path(sql_path)
            
            if sql_file.exists():
                return sql_file.read_text(encoding="utf-8")
        
        return ""
    
    def _create_augmenter_sql_snapshot(self) -> str:
        """Augmenter SQL 스냅샷 생성"""
        if not self.settings.model.augmenter:
            return ""
        
        augmenter_uri = self.settings.model.augmenter.source_uri
        
        # pass_through augmenter는 source_uri가 없을 수 있음
        if not augmenter_uri:
            return ""
        
        if augmenter_uri.startswith("bq://"):
            sql_path = augmenter_uri.replace("bq://", "")
            sql_file = Path(sql_path)
            
            if sql_file.exists():
                return sql_file.read_text(encoding="utf-8")
        
        return ""
    
    def _create_recipe_yaml_snapshot(self) -> str:
        """Recipe YAML 스냅샷 생성"""
        recipe_file = self.settings.model.computed["recipe_file"]
        recipe_path = Path(f"recipes/{recipe_file}.yaml")
        
        if recipe_path.exists():
            return recipe_path.read_text(encoding="utf-8")
        
        return ""
    
    def _create_training_metadata(self) -> Dict[str, Any]:
        """학습 메타데이터 생성"""
        from datetime import datetime
        
        return {
            "training_timestamp": datetime.now().isoformat(),
            "model_class": self.settings.model.computed["model_class_name"],
            "recipe_file": self.settings.model.computed["recipe_file"],
            "run_name": self.settings.model.computed["run_name"],
            "class_path": self.settings.model.class_path,
            "hyperparameters": self.settings.model.hyperparameters.root,
            "loader_uri": self.settings.model.loader.source_uri,
            "recipe_snapshot": self.settings.model.dict(),
        }
