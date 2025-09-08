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
from src.utils.system.console_manager import UnifiedConsole, get_console

if TYPE_CHECKING:
    from src.factory.artifact import PyfuncWrapper
    from src.interface import BaseFetcher


class Factory:
    """
    Recipe 설정(settings.recipe)에 기반하여 모든 핵심 컴포넌트를 생성하는 중앙 팩토리 클래스.
    일관된 접근 패턴과 캐싱을 통해 효율적인 컴포넌트 생성을 보장합니다.
    """
    # 클래스 변수: 컴포넌트 등록 상태 추적
    _components_registered: ClassVar[bool] = False
    
    def __init__(self, settings: Settings):
        # 컴포넌트 자동 등록 (최초 1회만)
        self._ensure_components_registered()
        
        self.settings = settings
        self.console = UnifiedConsole(settings)
        
        # Recipe 구조 검증
        if not self.settings.recipe:
            raise ValueError("Recipe 구조가 필요합니다. settings.recipe가 없습니다.")
        
        # 자주 사용하는 경로 캐싱 (일관된 접근 패턴)
        self._recipe = settings.recipe
        self._config = settings.config
        self._data = self._recipe.data
        self._model = self._recipe.model
        
        # 생성된 컴포넌트 캐싱
        self._component_cache: Dict[str, Any] = {}
        
        self.console.info(f"Factory initialized with Recipe: {self._recipe.name}",
                         rich_message=f"🏭 Factory initialized: [cyan]{self._recipe.name}[/cyan]")
    
    @classmethod
    def _ensure_components_registered(cls) -> None:
        """
        컴포넌트들이 Registry에 등록되었는지 확인하고, 필요시 등록합니다.
        이 메서드는 Factory 인스턴스가 처음 생성될 때 한 번만 실행됩니다.
        """
        if not cls._components_registered:
            # Use global console for classmethod
            console = get_console()
            console.info("Initializing component registries...", rich_message="🔧 Initializing component registries...")
            
            # 컴포넌트 모듈들을 import하여 self-registration 트리거
            try:
                import src.components.adapter
                import src.components.evaluator
                import src.components.fetcher
                import src.components.trainer
                import src.components.preprocessor
                import src.components.datahandler
            except ImportError as e:
                console.warning(f"Some components could not be imported: {e}")
            
            cls._components_registered = True
            console.info("Component registries initialized successfully", rich_message="✅ Component registries initialized")
    
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
            self.console.info(f"Created instance from class path: {class_path}", rich_message=f"✅ Created: [cyan]{class_path.split('.')[-1]}[/cyan]")
            return instance
            
        except Exception as e:
            self.console.error(f"Failed to create instance from {class_path}: {e}")
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
                    self.console.info(
                        f"Converted hyperparameter '{key}' to callable: {value}", 
                        rich_message=f"🔧 Converted param: [yellow]{key}[/yellow] → callable",
                        context={"module_path": module_path, "func_name": func_name}
                    )
                except (ImportError, AttributeError):
                    self.console.info(
                        f"Keeping hyperparameter '{key}' as string: {value}", 
                        rich_message=f"📝 Keeping param: [yellow]{key}[/yellow] as string",
                        context={"module_path": module_path, "func_name": func_name}
                    )
        
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
        self.console.warning(
            f"source_uri 패턴을 인식할 수 없습니다: {source_uri}. 'storage' 어댑터를 사용합니다.", 
            rich_message=f"⚠️ Unknown source_uri pattern: [red]{source_uri}[/red] → using 'storage' adapter"
        )
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
            self.console.info("캐시된 어댑터 반환", 
                            rich_message=f"🔄 Using cached adapter: [dim]{cache_key}[/dim]")
            return self._component_cache[cache_key]
        
        # 어댑터 타입 결정 (일관된 접근 패턴)
        if adapter_type:
            target_type = adapter_type
        else:
            # 1순위: config에서 명시된 adapter_type 사용
            config_adapter_type = getattr(self.settings.config.data_source, 'adapter_type', None)
            if config_adapter_type:
                target_type = config_adapter_type
                self.console.info(f"Using configured adapter type: {target_type}",
                                rich_message=f"⚙️ Using config adapter: [cyan]{target_type}[/cyan]")
            else:
                # 2순위: source_uri에서 자동 감지
                source_uri = self._data.loader.source_uri
                target_type = self._detect_adapter_type_from_uri(source_uri)
                self.console.info(f"Auto-detected adapter type '{target_type}' from URI: {source_uri}",
                                rich_message=f"🔍 Auto-detected adapter: [cyan]{target_type}[/cyan] from URI")
        
        # Registry를 통한 생성 (일관된 패턴)
        try:
            self.console.component_init(f"Data Adapter ({target_type})", "success")
            adapter = AdapterRegistry.create(target_type, self.settings)
            self._component_cache[cache_key] = adapter  # 캐싱
            self.console.info(f"Created data adapter: {target_type}",
                            rich_message=f"✅ Data adapter created: [green]{target_type}[/green]")
            return adapter
        except Exception as e:
            available = list(AdapterRegistry.list_adapters().keys())
            self.console.error(f"Failed to create adapter '{target_type}'", 
                             rich_message=f"❌ Adapter creation failed: [red]{target_type}[/red]",
                             context={"available_adapters": available, "target_type": target_type},
                             suggestion="Check adapter configuration and available adapters")
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
            self.console.info(f"캐시된 fetcher 반환: {mode}",
                            rich_message=f"🔄 Using cached fetcher: [dim]{mode}[/dim]")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴으로 설정 접근
        env = self._config.environment.name if hasattr(self._config, "environment") else "local"
        provider = self.settings.config.feature_store.provider if self.settings.config.feature_store else "none"
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
            fetcher_name = fetch_type or 'pass_through'
            self.console.component_init(f"Fetcher ({fetcher_name}, {mode})", "success")
            self.console.info(f"Created fetcher: {fetcher_name} (mode={mode})",
                            rich_message=f"✅ Fetcher created: [green]{fetcher_name}[/green] ([dim]{mode}[/dim])")
            return fetcher
            
        except Exception as e:
            self.console.error(f"Failed to create fetcher: {e}",
                             rich_message=f"❌ Fetcher creation failed: {e}",
                             context={"mode": mode, "env": env, "provider": provider, "fetch_type": fetch_type},
                             suggestion="Check fetcher configuration and feature store settings")
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
            self.console.info("캐시된 preprocessor 반환",
                            rich_message=f"🔄 Using cached preprocessor")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴
        preprocessor_config = getattr(self._recipe, "preprocessor", None)
        
        if not preprocessor_config:
            self.console.info("No preprocessor configured",
                            rich_message="ℹ️  No preprocessor configured, skipping")
            return None
        
        try:
            # Preprocessor 생성
            self.console.component_init("Preprocessor", "success")
            preprocessor = Preprocessor(settings=self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = preprocessor
            self.console.info("Created preprocessor",
                            rich_message="✅ Preprocessor created: [green]ready[/green]")
            return preprocessor
            
        except Exception as e:
            self.console.error(f"Failed to create preprocessor: {e}",
                             rich_message=f"❌ Preprocessor creation failed: {e}",
                             context={"config_available": bool(preprocessor_config)},
                             suggestion="Check preprocessor configuration in recipe")
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
            self.console.info("캐시된 model 반환",
                            rich_message=f"🔄 Using cached model")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴
        class_path = self._model.class_path
        
        # 하이퍼파라미터 추출 (tuning 메타데이터 제외)
        hyperparameters = {}
        if hasattr(self._model.hyperparameters, 'tuning_enabled'):
            if self._model.hyperparameters.tuning_enabled:
                # 튜닝 활성화시: fixed 파라미터만 사용
                if hasattr(self._model.hyperparameters, 'fixed') and self._model.hyperparameters.fixed:
                    hyperparameters = self._model.hyperparameters.fixed.copy()
            else:
                # 튜닝 비활성화시: values 파라미터 사용
                if hasattr(self._model.hyperparameters, 'values') and self._model.hyperparameters.values:
                    hyperparameters = self._model.hyperparameters.values.copy()
        else:
            # 레거시 구조: 전체 dict에서 tuning 메타데이터 제외
            hyperparameters = dict(self._model.hyperparameters) if hasattr(self._model.hyperparameters, '__dict__') else {}
            # 튜닝 관련 메타데이터 제거
            tuning_keys = ['tuning_enabled', 'optimization_metric', 'direction', 'n_trials', 'timeout', 'fixed', 'tunable', 'values']
            for key in tuning_keys:
                hyperparameters.pop(key, None)
        
        try:
            # 헬퍼 메서드 활용
            self.console.component_init(f"Model ({class_path.split('.')[-1]})", "success")
            model = self._create_from_class_path(class_path, hyperparameters)
            
            # 캐싱 저장
            self._component_cache[cache_key] = model
            self.console.info(f"Created model: {class_path}",
                            rich_message=f"✅ Model created: [green]{class_path.split('.')[-1]}[/green]")
            return model
            
        except Exception as e:
            self.console.error(f"Failed to create model from {class_path}: {e}",
                             rich_message=f"❌ Model creation failed: {e}",
                             context={"class_path": class_path, "hyperparams_count": len(hyperparameters)},
                             suggestion="Check model class path and hyperparameters")
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
            self.console.info("캐시된 evaluator 반환",
                            rich_message=f"🔄 Using cached evaluator")
            return self._component_cache[cache_key]
        
        # task_choice 활용
        task_choice = self._recipe.task_choice
        data_interface = self._recipe.data.data_interface
        
        try:
            # Registry 패턴으로 생성
            self.console.component_init(f"Evaluator ({task_choice})", "success")
            evaluator = EvaluatorRegistry.create(task_choice, self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = evaluator
            self.console.info(f"Created evaluator for task: {task_choice}",
                            rich_message=f"✅ Evaluator created: [green]{task_choice}[/green]")
            return evaluator
            
        except Exception as e:
            available = EvaluatorRegistry.get_available_tasks()
            self.console.error(f"Failed to create evaluator for '{task_choice}'", 
                             rich_message=f"❌ Evaluator creation failed: [red]{task_choice}[/red]",
                             context={"task_choice": task_choice, "available_evaluators": available},
                             suggestion="Check task choice and available evaluators")
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
            self.console.info(f"캐시된 trainer 반환: {trainer_type or 'default'}",
                            rich_message=f"🔄 Using cached trainer: [dim]{trainer_type or 'default'}[/dim]")
            return self._component_cache[cache_key]
        
        # TrainerRegistry import
        from src.components.trainer import TrainerRegistry
        
        # 일관된 접근 패턴
        trainer_type = trainer_type or 'default'
        
        try:
            # settings와 factory_provider를 전달하여 trainer 생성
            self.console.component_init(f"Trainer ({trainer_type})", "success")
            trainer = TrainerRegistry.create(
                trainer_type, 
                settings=self.settings,
                factory_provider=lambda: self
            )
            
            # 캐싱 저장
            self._component_cache[cache_key] = trainer
            self.console.info(f"Created trainer: {trainer_type}",
                            rich_message=f"✅ Trainer created: [green]{trainer_type}[/green]")
            return trainer
            
        except Exception as e:
            available = list(TrainerRegistry.trainers.keys())
            self.console.error(f"Failed to create trainer for '{trainer_type}'",
                             rich_message=f"❌ Trainer creation failed: [red]{trainer_type}[/red]",
                             context={"trainer_type": trainer_type, "available_trainers": available},
                             suggestion="Check trainer type and available trainers")
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
            self.console.info("캐시된 datahandler 반환",
                            rich_message=f"🔄 Using cached datahandler")
            return self._component_cache[cache_key]
        
        # DataHandlerRegistry import
        from src.components.datahandler import DataHandlerRegistry
        
        # task_choice 활용
        task_choice = self._recipe.task_choice
        
        try:
            # 모델 클래스 경로 추출 (catalog 기반 핸들러 선택을 위해)
            model_class_path = getattr(self._recipe.model, 'class_path', None)
            model_name = model_class_path.split('.')[-1] if model_class_path else 'unknown'
            
            # Registry 패턴으로 catalog 기반 핸들러 선택
            self.console.component_init(f"DataHandler ({task_choice}, {model_name})", "success")
            datahandler = DataHandlerRegistry.get_handler_for_task(
                task_choice, 
                self.settings, 
                model_class_path=model_class_path
            )
            
            # 캐싱 저장
            self._component_cache[cache_key] = datahandler
            self.console.info(f"Created datahandler for task: {task_choice}, model: {model_class_path}",
                            rich_message=f"✅ DataHandler created: [green]{task_choice}[/green] + [dim]{model_name}[/dim]")
            return datahandler
            
        except Exception as e:
            available = list(DataHandlerRegistry.get_available_handlers().keys())
            self.console.error(f"Failed to create datahandler for '{task_choice}'",
                             rich_message=f"❌ DataHandler creation failed: [red]{task_choice}[/red]",
                             context={"task_choice": task_choice, "model_class_path": model_class_path, "available_handlers": available},
                             suggestion="Check task choice, model path and available data handlers")
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
            self.console.info("캐시된 feature store adapter 반환",
                            rich_message=f"🔄 Using cached feature store adapter")
            return self._component_cache[cache_key]
        
        # 검증
        if not self.settings.config.feature_store:
            raise ValueError("Feature Store settings are not configured.")
        
        try:
            # Registry 패턴으로 생성
            self.console.component_init("Feature Store Adapter", "success")
            adapter = AdapterRegistry.create('feature_store', self.settings)
            
            # 캐싱 저장
            self._component_cache[cache_key] = adapter
            self.console.info("Created Feature Store adapter",
                            rich_message=f"✅ Feature Store adapter created: [green]ready[/green]")
            return adapter
            
        except Exception as e:
            self.console.error(f"Failed to create Feature Store adapter: {e}",
                             rich_message=f"❌ Feature Store adapter failed: {e}",
                             context={"feature_store_config": bool(self.settings.config.feature_store)},
                             suggestion="Check feature store configuration")
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
            self.console.info("캐시된 Optuna integration 반환",
                            rich_message=f"🔄 Using cached Optuna integration")
            return self._component_cache[cache_key]
        
        # 일관된 접근 패턴 (Recipe hyperparameters 구조 사용)
        tuning_config = getattr(self._model, "hyperparameters", None)
        
        if not tuning_config:
            raise ValueError("Hyperparameter tuning settings are not configured.")
        
        try:
            from src.utils.integrations.optuna_integration import OptunaIntegration
            
            # Integration 생성
            self.console.component_init("Optuna Integration", "success")
            integration = OptunaIntegration(tuning_config)
            
            # 캐싱 저장
            self._component_cache[cache_key] = integration
            self.console.info("Created Optuna integration",
                            rich_message=f"✅ Optuna integration created: [green]ready[/green]")
            return integration
            
        except ImportError as e:
            self.console.error("Optuna is not installed. Please install with 'pip install optuna'",
                             rich_message="❌ Optuna not installed",
                             suggestion="Install with: pip install optuna")
            raise
        except Exception as e:
            self.console.error(f"Failed to create Optuna integration: {e}",
                             rich_message=f"❌ Optuna integration failed: {e}",
                             context={"tuning_config_available": bool(tuning_config)},
                             suggestion="Check hyperparameter tuning configuration")
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
        """완전한 스키마 정보가 캡슐화된 Artifact 생성"""
        from src.factory.artifact import PyfuncWrapper
        self.console.info("Creating PyfuncWrapper artifact...",
                         rich_message="📦 Creating PyfuncWrapper artifact")
        
        signature, data_schema = None, None
        if training_df is not None:
            self.console.info("Generating model signature and data schema from training_df...",
                            rich_message="🔍 Generating model signature and schema")
            from src.utils.integrations.mlflow_integration import create_enhanced_model_signature_with_schema
            
            # 데이터 수집
            fetcher_conf = self._recipe.data.fetcher
            data_interface = self._recipe.data.data_interface
            
            # Timestamp 컬럼 처리
            ts_col = fetcher_conf.timestamp_column if fetcher_conf else None
            if ts_col and ts_col in training_df.columns:
                import pandas as pd
                if not pd.api.types.is_datetime64_any_dtype(training_df[ts_col]):
                    training_df = training_df.copy()
                    training_df[ts_col] = pd.to_datetime(training_df[ts_col], errors='coerce')

            # data_interface_config 구성
            data_interface_config = {
                'entity_columns': data_interface.entity_columns,
                'timestamp_column': ts_col,
                'task_type': self._recipe.task_choice,
                'target_column': data_interface.target_column,
                'treatment_column': getattr(data_interface, 'treatment_column', None),
            }
            
            signature, data_schema = create_enhanced_model_signature_with_schema(
                training_df, 
                data_interface_config
            )
            self.console.info("Signature and data schema created successfully.",
                            rich_message="✅ Signature and schema created successfully")
        
        # DataInterface 기반 검증용 스키마 생성
        data_interface_schema = None
        if training_df is not None:
            from src.utils.system.data_validation import create_data_interface_schema_for_storage
            data_interface_schema = create_data_interface_schema_for_storage(
                data_interface=self._recipe.data.data_interface,
                df=training_df,
                task_choice=self._recipe.task_choice
            )
            required_cols = len(data_interface_schema.get('required_columns', []))
            self.console.info(f"DataInterface 스키마 생성 완료: {required_cols}개 필수 컬럼",
                            rich_message=f"✅ DataInterface schema created: [cyan]{required_cols}[/cyan] required columns")
        
        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_datahandler=trained_datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
            data_interface_schema=data_interface_schema,
        )
