import yaml
import importlib
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import mlflow
import pandas as pd

from src.core.augmenter import Augmenter, LocalFileAugmenter, BaseAugmenter
from src.core.preprocessor import BasePreprocessor, Preprocessor
from src.interface.base_adapter import BaseAdapter
from src.settings.settings import Settings
from src.utils.system.logger import logger
from src.utils.adapters.file_system_adapter import FileSystemAdapter
from src.utils.adapters.bigquery_adapter import BigQueryAdapter
from src.utils.adapters.gcs_adapter import GCSAdapter
from src.utils.adapters.s3_adapter import S3Adapter

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

    def create_data_adapter(self, scheme: str) -> BaseAdapter:
        logger.info(f"'{scheme}' 스킴에 대한 데이터 어댑터를 생성합니다.")
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

    def create_augmenter(self) -> BaseAugmenter:
        """
        🆕 Blueprint v17.0: Feature Store 방식과 기존 SQL 방식 모두 지원하는 Augmenter 생성
        """
        augmenter_config = self.settings.model.augmenter
        if not augmenter_config:
            raise ValueError("Augmenter 설정이 레시피에 없습니다.")
        
        # 로컬 환경에서 local_override_uri가 있는 경우 (기존 방식 유지)
        is_local = self.settings.environment.app_env == "local"
        if is_local and hasattr(augmenter_config, 'local_override_uri') and augmenter_config.local_override_uri:
            logger.info("로컬 환경: LocalFileAugmenter 사용")
            return LocalFileAugmenter(uri=augmenter_config.local_override_uri)
        
        # 🆕 Feature Store 방식 체크
        if hasattr(augmenter_config, 'type') and augmenter_config.type == "feature_store":
            logger.info("🆕 Feature Store 방식 Augmenter 생성")
            return Augmenter(
                source_uri=None,
                settings=self.settings,
                augmenter_config=augmenter_config.dict()  # Pydantic 모델을 dict로 변환
            )
        else:
            # 🔄 기존 SQL 방식 (완전 호환성 유지)
            logger.info("🔄 기존 SQL 방식 Augmenter 생성")
            source_uri = augmenter_config.source_uri if hasattr(augmenter_config, 'source_uri') else None
            if not source_uri:
                raise ValueError("기존 SQL 방식 Augmenter에는 source_uri가 필요합니다.")
            
            return Augmenter(
                source_uri=source_uri,
                settings=self.settings,
                augmenter_config={'type': 'sql'}  # 기존 방식 명시
            )

    def create_preprocessor(self) -> Optional[BasePreprocessor]:
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
