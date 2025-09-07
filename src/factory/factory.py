from __future__ import annotations
import importlib
from typing import Optional, Dict, Any, TYPE_CHECKING, ClassVar

import pandas as pd

from src.components.fetcher import FetcherRegistry
from src.components.preprocessor import Preprocessor, BasePreprocessor
from src.components.adapter import AdapterRegistry
from src.components.evaluator import EvaluatorRegistry
from src.interface import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.factory.artifact import PyfuncWrapper
    from src.interface import BaseFetcher


class Factory:
    """
    현대화된 Recipe 설정(settings.recipe)에 기반하여 모든 핵심 컴포넌트를 생성하는 중앙 팩토리 클래스.
    일관된 접근 패턴과 캐싱을 통해 효율적인 컴포넌트 생성을 보장합니다.
    """
    # 클래스 변수: 컴포넌트 등록 상태 추적
    _components_registered: ClassVar[bool] = False
    
    def __init__(self, settings: Settings):
        # 컴포넌트 자동 등록 (최초 1회만)
        self._ensure_components_registered()
        
        self.settings = settings
        
        # 현대화된 Recipe 구조 검증
        if not self.settings.recipe:
            raise ValueError("현대화된 Recipe 구조가 필요합니다. settings.recipe가 없습니다.")
        
        # 자주 사용하는 경로 캐싱 (일관된 접근 패턴)
        self._recipe = settings.recipe
        self._config = settings.config
        self._data = self._recipe.data
        self._model = self._recipe.model
        
        # 생성된 컴포넌트 캐싱
        self._component_cache: Dict[str, Any] = {}
        
        logger.info(f"Factory initialized with Recipe: {self._recipe.name}")
    
    @classmethod
    def _ensure_components_registered(cls) -> None:
        """
        컴포넌트들이 Registry에 등록되었는지 확인하고, 필요시 등록합니다.
        이 메서드는 Factory 인스턴스가 처음 생성될 때 한 번만 실행됩니다.
        """
        if not cls._components_registered:
            logger.debug("Initializing component registries...")
            
            # 컴포넌트 모듈들을 import하여 self-registration 트리거
            try:
                import src.components.adapter
                import src.components.evaluator
                import src.components.fetcher
                import src.components.trainer
                import src.components.preprocessor
                import src.components.datahandler
            except ImportError as e:
                logger.warning(f"Some components could not be imported: {e}")
            
            cls._components_registered = True
            logger.debug("Component registries initialized successfully")
    
    def _create_from_class_path(self, class_path: str, hyperparameters: Dict[str, Any]) -> Any:
        """
        클래스 경로로부터 동적으로 객체를 생성하는 헬퍼 메서드.
        
        Args:
            class_path: 전체 클래스 경로 (예: 'sklearn.ensemble.RandomForestClassifier')
            hyperparameters: 클래스 초기화 파라미터
            
        Returns:
            생성된 객체 인스턴스
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # 하이퍼파라미터 전처리 (callable 파라미터 처리)
            processed_params = self._process_hyperparameters(hyperparameters)
            
            instance = model_class(**processed_params)
            logger.info(f"✅ Created instance from class path: {class_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance from {class_path}: {e}")
            raise ValueError(f"Could not load class: {class_path}") from e
    
    def _process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """하이퍼파라미터 전처리 (문자열을 객체로 변환 등)."""
        processed = params.copy()
        
        for key, value in processed.items():
            if isinstance(value, str) and "." in value and ("_fn" in key or "_class" in key):
                try:
                    module_path, func_name = value.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    processed[key] = getattr(module, func_name)
                    logger.debug(f"Converted hyperparameter '{key}' to callable: {value}")
                except (ImportError, AttributeError):
                    logger.debug(f"Keeping hyperparameter '{key}' as string: {value}")
        
        return processed

    def _detect_adapter_type_from_uri(self, source_uri: str) -> str:
        """
        source_uri 패턴을 분석하여 필요한 어댑터 타입을 자동으로 결정합니다.
        
        패턴:
        - .sql 파일 또는 SQL 쿼리 → 'sql'
        - .csv, .parquet, .json → 'storage'
        - s3://, gs://, az:// → 'storage'
        - bigquery:// → 'bigquery'
        """
        uri_lower = source_uri.lower()
        
        # SQL 패턴
        if uri_lower.endswith('.sql') or 'select' in uri_lower or 'from' in uri_lower:
            return 'sql'
        
        # BigQuery 패턴
        if uri_lower.startswith('bigquery://'):
            return 'bigquery'
        
        # Cloud Storage 패턴
        if any(uri_lower.startswith(prefix) for prefix in ['s3://', 'gs://', 'az://']):
            return 'storage'
        
        # File 패턴
        if any(uri_lower.endswith(ext) for ext in ['.csv', '.parquet', '.json', '.tsv']):
            return 'storage'
        
        # 기본값
        logger.warning(f"source_uri 패턴을 인식할 수 없습니다: {source_uri}. 'storage' 어댑터를 사용합니다.")
        return 'storage'
    
    def create_data_adapter(self, adapter_type: Optional[str] = None) -> "BaseAdapter":
        """
        데이터 어댑터 생성 (일관된 접근 패턴).
        
        Args:
            adapter_type: 명시적 어댑터 타입 (선택사항)
            
        Returns:
            BaseAdapter 인스턴스
        """
        # 캐싱 확인
        cache_key = f"adapter_{adapter_type}" if adapter_type else "adapter_auto"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached adapter: {cache_key}")
            return self._component_cache[cache_key]
        
        # 어댑터 타입 결정 (일관된 접근 패턴)
        if adapter_type:
            target_type = adapter_type
        else:
            # 캐싱된 경로 사용
            source_uri = self._data.loader.source_uri
            target_type = self._detect_adapter_type_from_uri(source_uri)
            logger.info(f"Auto-detected adapter type '{target_type}' from URI: {source_uri}")
        
        # Registry를 통한 생성 (일관된 패턴)
        try:
            adapter = AdapterRegistry.create(target_type, self.settings)
            self._component_cache[cache_key] = adapter  # 캐싱
            logger.info(f"✅ Created data adapter: {target_type}")
            return adapter
        except Exception as e:
            available = list(AdapterRegistry.list_adapters().keys())
            raise ValueError(
                f"Failed to create adapter '{target_type}'. Available: {available}"
            ) from e
    
    def create_fetcher(self, run_mode: Optional[str] = None) -> "BaseFetcher":
        """
        Fetcher 생성 (일관된 접근 패턴 + 캐싱).
        
        Args:
            run_mode: 실행 모드 (batch/serving)
            
        Returns:
            BaseFetcher 인스턴스
        """
        # 캐싱 키 생성
        mode = (run_mode or "batch").lower()
        cache_key = f"fetcher_{mode}"
        
        # 캐싱 확인
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached fetcher: {cache_key}")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴으로 설정 접근
        env = self._config.environment.env_name if hasattr(self._config, "environment") else "local"
        provider = self.settings.feature_store.provider if self.settings.feature_store else "none"
        fetch_conf = self._recipe.data.fetcher if hasattr(self._recipe.data, "fetcher") else None
        fetch_type = fetch_conf.type if fetch_conf else None

        # serving 모드 검증
        if mode == "serving":
            if fetch_type in (None, "pass_through") or provider in (None, "none"):
                raise TypeError(
                    "Serving 모드에서는 Feature Store 연결이 필요합니다. "
                    "pass_through 또는 feature_store 미구성은 허용되지 않습니다."
                )

        # Fetcher 생성 (일관된 Registry 패턴)
        try:
            # PassThrough 케이스
            if env == "local" or provider == "none" or fetch_type == "pass_through" or not fetch_conf:
                fetcher = FetcherRegistry.create("pass_through")
                
            # Feature Store 케이스
            elif fetch_type == "feature_store" and provider in {"feast", "mock", "dynamic"}:
                fetcher = FetcherRegistry.create(
                    fetch_type, 
                    settings=self.settings, 
                    factory=self
                )
            else:
                raise ValueError(
                    f"적절한 fetcher를 선택할 수 없습니다. "
                    f"env={env}, provider={provider}, fetch_type={fetch_type}, mode={mode}"
                )
            
            # 캐싱 저장
            self._component_cache[cache_key] = fetcher
            logger.info(f"✅ Created fetcher: {fetch_type or 'pass_through'} (mode={mode})")
            return fetcher
            
        except Exception as e:
            logger.error(f"Failed to create fetcher: {e}")
            raise

    def create_preprocessor(self) -> Optional[BasePreprocessor]:
        """
        Preprocessor 생성 (일관된 접근 패턴 + 캐싱).
        
        Returns:
            BasePreprocessor 인스턴스 또는 None
        """
        # 캐싱 확인
        cache_key = "preprocessor"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached preprocessor")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴
        preprocessor_config = getattr(self._recipe, "preprocessor", None)
        
        if not preprocessor_config:
            logger.info("No preprocessor configured")
            return None
        
        try:
            # Preprocessor 생성
            preprocessor = Preprocessor(settings=self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = preprocessor
            logger.info("✅ Created preprocessor")
            return preprocessor
            
        except Exception as e:
            logger.error(f"Failed to create preprocessor: {e}")
            raise

    def create_model(self) -> Any:
        """
        Model 생성 (일관된 접근 패턴 + 헬퍼 메서드 활용).
        
        Returns:
            모델 인스턴스
        """
        # 캐싱 확인
        cache_key = "model"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached model")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴
        class_path = self._model.class_path
        
        # 하이퍼파라미터 추출 (일관된 패턴)
        if hasattr(self._model.hyperparameters, 'get_fixed_params'):
            hyperparameters = self._model.hyperparameters.get_fixed_params()
        else:
            hyperparameters = dict(self._model.hyperparameters) if hasattr(self._model.hyperparameters, '__dict__') else {}
        
        try:
            # 헬퍼 메서드 활용
            model = self._create_from_class_path(class_path, hyperparameters)
            
            # 캐싱 저장
            self._component_cache[cache_key] = model
            logger.info(f"✅ Created model: {class_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model from {class_path}: {e}")
            raise

    def create_evaluator(self) -> Any:
        """
        Evaluator 생성 (일관된 접근 패턴 + 캐싱).
        
        Returns:
            Evaluator 인스턴스
        """
        # 캐싱 확인
        cache_key = "evaluator"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached evaluator")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴
        data_interface = self._recipe.data.data_interface
        task_type = data_interface.task_type
        
        try:
            # Registry 패턴으로 생성
            evaluator = EvaluatorRegistry.create(task_type, data_interface)
            
            # 캐싱 저장
            self._component_cache[cache_key] = evaluator
            logger.info(f"✅ Created evaluator for task: {task_type}")
            return evaluator
            
        except Exception as e:
            available = list(EvaluatorRegistry.list_evaluators().keys())
            logger.error(f"Failed to create evaluator for '{task_type}'. Available: {available}")
            raise

    def create_trainer(self, trainer_type: Optional[str] = None) -> Any:
        """
        Trainer 생성 (일관된 접근 패턴 + 캐싱).
        
        Args:
            trainer_type: 트레이너 타입 (None이면 'default' 사용)
            
        Returns:
            Trainer 인스턴스
        """
        # 캐싱 확인
        cache_key = f"trainer_{trainer_type or 'default'}"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached trainer")
            return self._component_cache[cache_key]
        
        # TrainerRegistry import
        from src.components.trainer import TrainerRegistry
        
        # 일관된 접근 패턴
        trainer_type = trainer_type or 'default'
        
        try:
            # settings와 factory_provider를 전달하여 trainer 생성
            trainer = TrainerRegistry.create(
                trainer_type, 
                settings=self._settings,
                factory_provider=lambda: self
            )
            
            # 캐싱 저장
            self._component_cache[cache_key] = trainer
            logger.info(f"✅ Created trainer: {trainer_type}")
            return trainer
            
        except Exception as e:
            available = list(TrainerRegistry.trainers.keys())
            logger.error(f"Failed to create trainer for '{trainer_type}'. Available: {available}")
            raise

    def create_datahandler(self) -> Any:
        """
        DataHandler 생성 (일관된 접근 패턴 + 캐싱).
        task_type에 따라 적절한 DataHandler를 자동으로 선택합니다.
        
        Returns:
            BaseDataHandler 인스턴스
        """
        # 캐싱 확인
        cache_key = "datahandler"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached datahandler")
            return self._component_cache[cache_key]
        
        # DataHandlerRegistry import
        from src.components.datahandler import DataHandlerRegistry
        
        # 일관된 접근 패턴
        data_interface = self._recipe.data.data_interface
        task_type = data_interface.task_type
        
        try:
            # Registry 패턴으로 task_type에 따라 자동 생성
            datahandler = DataHandlerRegistry.get_handler_for_task(task_type, self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = datahandler
            logger.info(f"✅ Created datahandler for task: {task_type}")
            return datahandler
            
        except Exception as e:
            available = list(DataHandlerRegistry.get_available_handlers().keys())
            logger.error(f"Failed to create datahandler for '{task_type}'. Available: {available}")
            raise

    def create_feature_store_adapter(self) -> "BaseAdapter":
        """
        Feature Store 어댑터 생성 (일관된 접근 패턴 + 캐싱).
        
        Returns:
            Feature Store 어댑터 인스턴스
        """
        # 캐싱 확인
        cache_key = "feature_store_adapter"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached feature store adapter")
            return self._component_cache[cache_key]
        
        # 검증
        if not self.settings.feature_store:
            raise ValueError("Feature Store settings are not configured.")
        
        try:
            # Registry 패턴으로 생성
            adapter = AdapterRegistry.create('feature_store', self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = adapter
            logger.info("✅ Created Feature Store adapter")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create Feature Store adapter: {e}")
            raise
    
    def create_optuna_integration(self) -> Any:
        """
        Optuna Integration 생성 (일관된 접근 패턴 + 캐싱).
        
        Returns:
            OptunaIntegration 인스턴스
        """
        # 캐싱 확인
        cache_key = "optuna_integration"
        if cache_key in self._component_cache:
            logger.debug(f"Returning cached Optuna integration")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴 (Recipe hyperparameters 구조 사용)
        tuning_config = getattr(self._model, "hyperparameters", None)
        
        if not tuning_config:
            raise ValueError("Hyperparameter tuning settings are not configured.")
        
        try:
            from src.utils.integrations.optuna_integration import OptunaIntegration
            
            # Integration 생성
            integration = OptunaIntegration(tuning_config)
            
            # 캐싱 저장
            self._component_cache[cache_key] = integration
            logger.info("✅ Created Optuna integration")
            return integration
            
        except ImportError as e:
            logger.error("Optuna is not installed. Please install with 'pip install optuna'")
            raise
        except Exception as e:
            logger.error(f"Failed to create Optuna integration: {e}")
            raise

    def create_pyfunc_wrapper(
        self, 
        trained_model: Any, 
        trained_datahandler: Any,
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional['BaseFetcher'],
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None
    ) -> PyfuncWrapper:
        """🔄 Phase 5: 완전한 스키마 정보가 캡슐화된 Enhanced Artifact 생성"""
        from src.factory.artifact import PyfuncWrapper
        logger.info("Creating PyfuncWrapper artifact...")
        
        signature, data_schema = None, None
        if training_df is not None:
            logger.info("Generating model signature and data schema from training_df...")
            from src.utils.integrations.mlflow_integration import create_enhanced_model_signature_with_schema
            
            # ✅ 새로운 구조에서 데이터 수집
            fetcher_conf = self._recipe.data.fetcher
            data_interface = self._recipe.data.data_interface
            
            # Timestamp 컬럼 처리
            ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
            if ts_col and ts_col in training_df.columns:
                import pandas as pd
                if not pd.api.types.is_datetime64_any_dtype(training_df[ts_col]):
                    training_df = training_df.copy()
                    training_df[ts_col] = pd.to_datetime(training_df[ts_col], errors='coerce')

            # ✅ 새로운 구조로 data_interface_config 구성
            data_interface_config = {
                'entity_columns': data_interface.entity_columns,
                'timestamp_column': ts_col,
                'task_type': data_interface.task_type,
                'target_column': data_interface.target_column,
                'treatment_column': getattr(data_interface, 'treatment_column', None),
            }
            
            signature, data_schema = create_enhanced_model_signature_with_schema(
                training_df, 
                data_interface_config
            )
            logger.info("✅ Signature and data schema created successfully.")
        
        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_datahandler=trained_datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
        )
