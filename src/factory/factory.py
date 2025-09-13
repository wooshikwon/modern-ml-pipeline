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
from src.utils.core.console_manager import UnifiedConsole, get_console

if TYPE_CHECKING:
    from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper
    from src.interface import BaseFetcher


class CalibrationEvaluatorWrapper:
    """
    Calibration 평가 로직을 캡슐화한 Wrapper 클래스
    모든 복잡한 분기 처리와 평가 로직을 담당
    """

    def __init__(self, trained_model, trained_calibrator, console):
        self.trained_model = trained_model
        self.trained_calibrator = trained_calibrator
        self.console = console

    def evaluate(self, X_test, y_test) -> dict:
        """
        Calibration 평가 수행 (모든 복잡한 로직 처리)

        Args:
            X_test: 테스트 특성
            y_test: 테스트 라벨

        Returns:
            Calibration metrics 딕셔너리
        """
        from src.components.calibration.evaluator import evaluate_calibration_metrics, evaluate_multiclass_calibration

        # Uncalibrated 확률 얻기
        y_prob_uncalibrated = self.trained_model.predict_proba(X_test)
        y_prob_calibrated = self.trained_calibrator.transform(y_prob_uncalibrated)

        # Binary vs Multiclass 자동 구분 및 평가
        if y_prob_uncalibrated.ndim == 2 and y_prob_uncalibrated.shape[1] == 2:
            # Binary classification - positive class 확률만 사용
            calibration_metrics = evaluate_calibration_metrics(
                y_test,
                y_prob_uncalibrated[:, 1],
                y_prob_calibrated[:, 1] if y_prob_calibrated.ndim == 2 else y_prob_calibrated
            )
            self.console.info("Binary calibration 평가를 완료했습니다",
                            rich_message="📊 Binary calibration 평가를 완료했습니다")

        elif y_prob_uncalibrated.ndim == 2 and y_prob_uncalibrated.shape[1] > 2:
            # Multiclass classification
            calibration_metrics = evaluate_multiclass_calibration(
                y_test, y_prob_uncalibrated, y_prob_calibrated
            )
            self.console.info(f"Multiclass calibration 평가를 완료했습니다 ({y_prob_uncalibrated.shape[1]}개 클래스)",
                            rich_message=f"📊 Multiclass calibration 평가를 완료했습니다: [cyan]{y_prob_uncalibrated.shape[1]}[/cyan]개 클래스")

        else:
            # 1D case (이미 binary positive class만 있는 경우)
            calibration_metrics = evaluate_calibration_metrics(
                y_test, y_prob_uncalibrated, y_prob_calibrated
            )
            self.console.info("1D calibration 평가를 완료했습니다",
                            rich_message="📊 1D calibration 평가를 완료했습니다")

        # Nested dict 제거 (MLflow 로깅을 위해)
        flat_metrics = {}
        for key, value in calibration_metrics.items():
            if not isinstance(value, dict):  # class_metrics 같은 nested dict 제외
                flat_metrics[f"calibration_{key}"] = value

        return flat_metrics


class Factory:
    """
    3-Tier Component Architecture를 통한 MLOps 컴포넌트 중앙 팩토리 클래스.
    Recipe 설정(settings.recipe)에 기반하여 계층별 패턴에 따라 컴포넌트를 생성합니다.

    Architecture Overview:
    =====================

    🔹 Tier 1: Atomic Components (원자적 컴포넌트)
       - Registry Pattern: XXXRegistry.create()
       - 단일 책임, 독립적 기능
       - 예: Evaluator, Fetcher, DataAdapter, Calibrator

    🔹 Tier 2: Composite Components (조합형 컴포넌트)
       - Factory-aware Registry Pattern: XXXRegistry.create(..., factory_provider=self)
       - 다른 컴포넌트들에 대한 의존성 필요
       - 예: Trainer, DataHandler

    🔹 Tier 3: Orchestrator Components (오케스트레이션 컴포넌트)
       - Direct Instantiation Pattern: Class(settings=self.settings)
       - 복수의 하위 컴포넌트들을 동적 조합/관리
       - 예: Preprocessor (다수 전처리 스텝 조합)

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

        self.console.info(f"Factory를 Recipe와 함께 초기화했습니다: {self._recipe.name}",
                         rich_message=f"🏭 Factory를 초기화했습니다: [cyan]{self._recipe.name}[/cyan]")

    @classmethod
    def _ensure_components_registered(cls) -> None:
        """
        컴포넌트들이 Registry에 등록되었는지 확인하고, 필요시 등록합니다.
        이 메서드는 Factory 인스턴스가 처음 생성될 때 한 번만 실행됩니다.
        """
        if not cls._components_registered:
            # Use global console for classmethod
            console = get_console()
            console.info("Component registry를 초기화하는 중...", rich_message="🔧 Component registry를 초기화하는 중...")

            # 컴포넌트 모듈들을 import하여 self-registration 트리거
            try:
                import src.components.adapter
                import src.components.evaluator
                import src.components.fetcher
                import src.components.trainer
                import src.components.preprocessor
                import src.components.datahandler
            except ImportError as e:
                console.warning(f"일부 component를 가져오지 못했습니다: {e}")

            cls._components_registered = True
            console.info("Component registry 초기화를 완료했습니다", rich_message="✅ Component registry 초기화 완료")

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
            self.console.info(f"클래스 경로에서 인스턴스를 생성했습니다: {class_path}", rich_message=f"✅ 생성 완료: [cyan]{class_path.split('.')[-1]}[/cyan]")
            return instance

        except Exception as e:
            self.console.error(f"{class_path}에서 인스턴스 생성에 실패했습니다: {e}")
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
                        f"Hyperparameter '{key}'를 callable로 변환했습니다: {value}",
                        rich_message=f"🔧 Parameter 변환: [yellow]{key}[/yellow] → callable",
                        context={"module_path": module_path, "func_name": func_name}
                    )
                except (ImportError, AttributeError):
                    self.console.info(
                        f"Hyperparameter '{key}'를 문자열로 유지합니다: {value}",
                        rich_message=f"📝 Parameter 유지: [yellow]{key}[/yellow] as string",
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
        - bigquery:// → 'sql'
        """
        uri_lower = source_uri.lower()

        # SQL 패턴
        if uri_lower.endswith('.sql') or 'select' in uri_lower or 'from' in uri_lower:
            return 'sql'

        # BigQuery 패턴 → SQL adapter로 통합
        if uri_lower.startswith('bigquery://'):
            return 'sql'  # 'bigquery' 대신 'sql' 반환

        # Cloud Storage 패턴
        if any(uri_lower.startswith(prefix) for prefix in ['s3://', 'gs://', 'az://']):
            return 'storage'

        # File 패턴
        if any(uri_lower.endswith(ext) for ext in ['.csv', '.parquet', '.json', '.tsv']):
            return 'storage'

        # 기본값
        self.console.warning(
            f"source_uri 패턴을 인식할 수 없습니다: {source_uri}. 'storage' 어댑터를 사용합니다.",
            rich_message=f"⚠️ 알 수 없는 source_uri 패턴: [red]{source_uri}[/red] → 'storage' adapter 사용"
        )
        return 'storage'

    # ===============================
    # Tier 1: Atomic Components
    # Registry Pattern - 단일 책임 컴포넌트
    # ===============================

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
                self.console.info(f"설정된 adapter 유형을 사용합니다: {target_type}",
                                rich_message=f"⚙️ 설정된 adapter 사용: [cyan]{target_type}[/cyan]")
            else:
                # 2순위: source_uri에서 자동 감지
                source_uri = self._data.loader.source_uri
                target_type = self._detect_adapter_type_from_uri(source_uri)
                self.console.info(f"URI에서 adapter 유형을 자동 감지했습니다: '{target_type}' from {source_uri}",
                                rich_message=f"🔍 Adapter 자동 감지: [cyan]{target_type}[/cyan] from URI")

        # Registry를 통한 생성 (일관된 패턴)
        try:
            self.console.component_init(f"Data Adapter ({target_type})", "success")
            adapter = AdapterRegistry.create(target_type, self.settings)
            self._component_cache[cache_key] = adapter  # 캐싱
            self.console.info(f"Data adapter를 생성했습니다: {target_type}",
                            rich_message=f"✅ Data adapter 생성 완료: [green]{target_type}[/green]")
            return adapter
        except Exception as e:
            available = list(AdapterRegistry.list_adapters().keys())
            self.console.error(f"'{target_type}' adapter 생성에 실패했습니다",
                             rich_message=f"❌ Adapter 생성 실패: [red]{target_type}[/red]",
                             context={"available_adapters": available, "target_type": target_type},
                             suggestion="Adapter 설정과 사용 가능한 adapter들을 확인해주세요")
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
            self.console.info(f"Fetcher를 생성했습니다: {fetcher_name} (mode={mode})",
                            rich_message=f"✅ Fetcher 생성 완료: [green]{fetcher_name}[/green] ([dim]{mode}[/dim])")
            return fetcher

        except Exception as e:
            self.console.error(f"Fetcher 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Fetcher 생성 실패: {e}",
                             context={"mode": mode, "env": env, "provider": provider, "fetch_type": fetch_type},
                             suggestion="Fetcher 설정과 Feature Store 설정을 확인해주세요")
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
                            rich_message="🔄 Using cached evaluator")
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
            self.console.info(f"Task에 대한 Evaluator를 생성했습니다: {task_choice}",
                            rich_message=f"✅ Evaluator 생성 완료: [green]{task_choice}[/green]")
            return evaluator

        except Exception:
            available = EvaluatorRegistry.get_available_tasks()
            self.console.error(f"'{task_choice}'에 대한 Evaluator 생성에 실패했습니다",
                             rich_message=f"❌ Evaluator 생성 실패: [red]{task_choice}[/red]",
                             context={"task_choice": task_choice, "available_evaluators": available},
                             suggestion="Task choice와 사용 가능한 evaluator들을 확인해주세요")
            raise

    def create_calibrator(self, method: Optional[str] = None) -> Optional[Any]:
        """
        Calibrator 생성 (조건에 따른 생성 분기 처리)

        Args:
            method: 캘리브레이션 방법 ('platt', 'isotonic' 등)

        Returns:
            BaseCalibrator 인스턴스 또는 None
        """
        # 캐싱 확인
        cache_key = f"calibrator_{method or 'default'}"
        if cache_key in self._component_cache:
            self.console.info("캐시된 calibrator 반환",
                            rich_message=f"🔄 Using cached calibrator: [cyan]{method}[/cyan]")
            return self._component_cache[cache_key]

        # Task와 calibration 설정 확인
        task_type = self._recipe.task_choice
        calibration_config = getattr(self._recipe.model, 'calibration', None)

        if not calibration_config or not getattr(calibration_config, 'enabled', False):
            self.console.info("Calibration이 비활성화되어 None을 반환합니다",
                            rich_message="🎯 Calibration: [dim]비활성화[/dim]")
            return None

        if task_type != 'classification':
            self.console.info(f"{task_type} task에서는 Calibration이 지원되지 않습니다",
                            rich_message=f"🎯 Calibration: [yellow]{task_type}에서 지원 안됨[/yellow]")
            return None

        # Method 결정 (기본값 없음 - Recipe에서 필수로 설정)
        calibration_method = method or getattr(calibration_config, 'method', None)
        if not calibration_method:
            raise ValueError(
                "Calibration method가 설정되지 않았습니다. "
                "Recipe에서 model.calibration.method를 설정하세요. "
                "사용 가능: 'beta', 'isotonic', 'temperature'"
            )

        try:
            from src.components.calibration.registry import CalibrationRegistry

            # Calibrator 생성
            calibrator = CalibrationRegistry.create(calibration_method)

            # 캐싱 저장
            self._component_cache[cache_key] = calibrator
            self.console.component_init(f"Calibrator ({calibration_method})", "success")
            self.console.info(f"Calibrator를 생성했습니다: {calibration_method}",
                            rich_message=f"✅ Calibrator 생성 완료: [green]{calibration_method}[/green]")
            return calibrator

        except Exception:
            from src.components.calibration.registry import CalibrationRegistry
            available = CalibrationRegistry.get_available_methods()
            self.console.error(f"'{calibration_method}' Calibrator 생성에 실패했습니다",
                             rich_message=f"❌ Calibrator 생성 실패: [red]{calibration_method}[/red]",
                             context={"method": calibration_method, "available_methods": available},
                             suggestion="Calibration method와 사용 가능한 calibrator들을 확인해주세요")
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
                            rich_message="🔄 Using cached feature store adapter")
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
            self.console.info("Feature Store adapter를 생성했습니다",
                            rich_message="✅ Feature Store adapter 생성 완료: [green]준비완료[/green]")
            return adapter

        except Exception as e:
            self.console.error(f"Feature Store adapter 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Feature Store adapter 실패: {e}",
                             context={"feature_store_config": bool(self.settings.config.feature_store)},
                             suggestion="Feature store 설정을 확인해주세요")
            raise

    # ===============================
    # Tier 2: Composite Components
    # Factory-aware Registry Pattern - 의존성 주입이 필요한 컴포넌트
    # ===============================

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
            self.console.info(f"Trainer를 생성했습니다: {trainer_type}",
                            rich_message=f"✅ Trainer 생성 완룼: [green]{trainer_type}[/green]")
            return trainer

        except Exception:
            available = list(TrainerRegistry.trainers.keys())
            self.console.error(f"'{trainer_type}' Trainer 생성에 실패했습니다",
                             rich_message=f"❌ Trainer 생성 실패: [red]{trainer_type}[/red]",
                             context={"trainer_type": trainer_type, "available_trainers": available},
                             suggestion="Trainer 유형과 사용 가능한 trainer들을 확인해주세요")
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
                            rich_message="🔄 Using cached datahandler")
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
            self.console.info(f"Task에 대한 DataHandler를 생성했습니다: {task_choice}, model: {model_class_path}",
                            rich_message=f"✅ DataHandler 생성 완료: [green]{task_choice}[/green] + [dim]{model_name}[/dim]")
            return datahandler

        except Exception:
            available = list(DataHandlerRegistry.get_available_handlers().keys())
            self.console.error(f"'{task_choice}'에 대한 DataHandler 생성에 실패했습니다",
                             rich_message=f"❌ DataHandler 생성 실패: [red]{task_choice}[/red]",
                             context={"task_choice": task_choice, "model_class_path": model_class_path, "available_handlers": available},
                             suggestion="Task choice, model path 및 사용 가능한 data handler들을 확인해주세요")
            raise

    # ===============================
    # Tier 3: Orchestrator Components
    # Direct Instantiation Pattern - 복잡한 조합 로직을 내장한 컴포넌트
    # ===============================

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
                            rich_message="🔄 Using cached preprocessor")
            return self._component_cache[cache_key]

        # 일관된 접근 패턴
        preprocessor_config = getattr(self._recipe, "preprocessor", None)

        if not preprocessor_config:
            self.console.info("Preprocessor가 구성되지 않았습니다",
                            rich_message="ℹ️  Preprocessor 미구성으로 건너뜁니다")
            return None

        try:
            # Preprocessor 생성
            self.console.component_init("Preprocessor", "success")
            preprocessor = Preprocessor(settings=self.settings)

            # 캐싱 저장
            self._component_cache[cache_key] = preprocessor
            self.console.info("Preprocessor를 생성했습니다",
                            rich_message="✅ Preprocessor 생성 완료: [green]준비완료[/green]")
            return preprocessor

        except Exception as e:
            self.console.error(f"Preprocessor 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Preprocessor 생성 실패: {e}",
                             context={"config_available": bool(preprocessor_config)},
                             suggestion="Recipe에서 Preprocessor 설정을 확인해주세요")
            raise

    # ===============================
    # Specialized Creation Methods
    # Recipe 구조 기반 동적 생성 - 설정에 따른 조건부 생성
    # ===============================

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
                            rich_message="🔄 Using cached model")
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
            self.console.info(f"Model을 생성했습니다: {class_path}",
                            rich_message=f"✅ Model 생성 완료: [green]{class_path.split('.')[-1]}[/green]")
            return model

        except Exception as e:
            self.console.error(f"{class_path}에서 Model 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Model 생성 실패: {e}",
                             context={"class_path": class_path, "hyperparams_count": len(hyperparameters)},
                             suggestion="Model 클래스 경로와 하이퍼파라미터를 확인해주세요")
            raise

    def create_optuna_integration(self) -> Any:
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
            self.console.info(f"Trainer를 생성했습니다: {trainer_type}",
                            rich_message=f"✅ Trainer 생성 완룼: [green]{trainer_type}[/green]")
            return trainer

        except Exception:
            available = list(TrainerRegistry.trainers.keys())
            self.console.error(f"'{trainer_type}' Trainer 생성에 실패했습니다",
                             rich_message=f"❌ Trainer 생성 실패: [red]{trainer_type}[/red]",
                             context={"trainer_type": trainer_type, "available_trainers": available},
                             suggestion="Trainer 유형과 사용 가능한 trainer들을 확인해주세요")
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
                            rich_message="🔄 Using cached datahandler")
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
            self.console.info(f"Task에 대한 DataHandler를 생성했습니다: {task_choice}, model: {model_class_path}",
                            rich_message=f"✅ DataHandler 생성 완료: [green]{task_choice}[/green] + [dim]{model_name}[/dim]")
            return datahandler

        except Exception:
            available = list(DataHandlerRegistry.get_available_handlers().keys())
            self.console.error(f"'{task_choice}'에 대한 DataHandler 생성에 실패했습니다",
                             rich_message=f"❌ DataHandler 생성 실패: [red]{task_choice}[/red]",
                             context={"task_choice": task_choice, "model_class_path": model_class_path, "available_handlers": available},
                             suggestion="Task choice, model path 및 사용 가능한 data handler들을 확인해주세요")
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
                            rich_message="🔄 Using cached feature store adapter")
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
            self.console.info("Feature Store adapter를 생성했습니다",
                            rich_message="✅ Feature Store adapter 생성 완료: [green]준비완료[/green]")
            return adapter

        except Exception as e:
            self.console.error(f"Feature Store adapter 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Feature Store adapter 실패: {e}",
                             context={"feature_store_config": bool(self.settings.config.feature_store)},
                             suggestion="Feature store 설정을 확인해주세요")
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
                            rich_message="🔄 Using cached Optuna integration")
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
            self.console.info("Optuna integration을 생성했습니다",
                            rich_message="✅ Optuna integration 생성 완료: [green]준비완료[/green]")
            return integration

        except ImportError:
            self.console.error("Optuna가 설치되지 않았습니다. 'pip install optuna'로 설치해주세요",
                             rich_message="❌ Optuna 미설치",
                             suggestion="pip install optuna로 설치해주세요")
            raise
        except Exception as e:
            self.console.error(f"Optuna integration 생성에 실패했습니다: {e}",
                             rich_message=f"❌ Optuna integration 실패: {e}",
                             context={"tuning_config_available": bool(tuning_config)},
                             suggestion="하이퍼파라미터 튜닝 설정을 확인해주세요")
            raise

    def create_calibrator(self, method: Optional[str] = None) -> Optional[Any]:
        """
        Calibrator 생성 (조건에 따른 생성 분기 처리)

        Args:
            method: 캘리브레이션 방법 ('platt', 'isotonic' 등)

        Returns:
            BaseCalibrator 인스턴스 또는 None
        """
        # 캐싱 확인
        cache_key = f"calibrator_{method or 'default'}"
        if cache_key in self._component_cache:
            self.console.info("캐시된 calibrator 반환",
                            rich_message=f"🔄 Using cached calibrator: [cyan]{method}[/cyan]")
            return self._component_cache[cache_key]

        # Task와 calibration 설정 확인
        task_type = self._recipe.task_choice
        calibration_config = getattr(self._recipe.model, 'calibration', None)

        if not calibration_config or not getattr(calibration_config, 'enabled', False):
            self.console.info("Calibration이 비활성화되어 None을 반환합니다",
                            rich_message="🎯 Calibration: [dim]비활성화[/dim]")
            return None

        if task_type != 'classification':
            self.console.info(f"{task_type} task에서는 Calibration이 지원되지 않습니다",
                            rich_message=f"🎯 Calibration: [yellow]{task_type}에서 지원 안됨[/yellow]")
            return None

        # Method 결정 (기본값 없음 - Recipe에서 필수로 설정)
        calibration_method = method or getattr(calibration_config, 'method', None)
        if not calibration_method:
            raise ValueError(
                "Calibration method가 설정되지 않았습니다. "
                "Recipe에서 model.calibration.method를 설정하세요. "
                "사용 가능: 'beta', 'isotonic', 'temperature'"
            )

        try:
            from src.components.calibration.registry import CalibrationRegistry

            # Calibrator 생성
            calibrator = CalibrationRegistry.create(calibration_method)

            # 캐싱 저장
            self._component_cache[cache_key] = calibrator
            self.console.component_init(f"Calibrator ({calibration_method})", "success")
            self.console.info(f"Calibrator를 생성했습니다: {calibration_method}",
                            rich_message=f"✅ Calibrator 생성 완료: [green]{calibration_method}[/green]")
            return calibrator

        except Exception:
            from src.components.calibration.registry import CalibrationRegistry
            available = CalibrationRegistry.get_available_methods()
            self.console.error(f"'{calibration_method}' Calibrator 생성에 실패했습니다",
                             rich_message=f"❌ Calibrator 생성 실패: [red]{calibration_method}[/red]",
                             context={"method": calibration_method, "available_methods": available},
                             suggestion="Calibration method와 사용 가능한 calibrator들을 확인해주세요")
            raise

    def create_calibration_evaluator(self, trained_model, trained_calibrator) -> Optional[Any]:
        """
        Calibration Evaluator 생성 및 실행 (모든 복잡한 로직 처리)

        Args:
            trained_model: 학습된 모델
            trained_calibrator: 학습된 calibrator

        Returns:
            Calibration metrics 또는 None
        """
        # Task와 calibrator 확인
        task_type = self._recipe.task_choice
        if task_type != 'classification' or not trained_calibrator:
            return None

        # 모델이 predict_proba를 지원하는지 확인
        if not hasattr(trained_model, 'predict_proba'):
            self.console.warning("Model이 predict_proba를 지원하지 않아 calibration 평가를 건너뜁니다",
                               rich_message="⚠️ predict_proba 지원하지 않음")
            return None

        return CalibrationEvaluatorWrapper(trained_model, trained_calibrator, self.console)

    def create_pyfunc_wrapper(
        self,
        trained_model: Any,
        trained_datahandler: Any,
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional['BaseFetcher'],
        trained_calibrator: Optional[Any] = None,
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None
    ) -> PyfuncWrapper:
        """완전한 스키마 정보가 캡슐화된 Artifact 생성"""
        from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper
        self.console.info("PyfuncWrapper artifact를 생성하는 중...",
                         rich_message="📦 PyfuncWrapper artifact 생성 중")

        signature, data_schema = None, None
        if training_df is not None:
            self.console.info("training_df에서 model signature와 data schema를 생성하는 중...",
                            rich_message="🔍 Model signature와 schema 생성 중")
            from src.utils.integrations.mlflow_integration import create_enhanced_model_signature_with_schema

            # 데이터 수집
            fetcher_conf = self._recipe.data.fetcher
            data_interface = self._recipe.data.data_interface

            # Timestamp 컬럼 처리 (fetcher → data_interface 순으로 폴백)
            ts_col = None
            if fetcher_conf and getattr(fetcher_conf, 'timestamp_column', None):
                ts_col = fetcher_conf.timestamp_column
            elif getattr(data_interface, 'timestamp_column', None):
                ts_col = data_interface.timestamp_column
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
            self.console.info("Signature와 data schema를 성공적으로 생성했습니다.",
                            rich_message="✅ Signature와 schema 생성 완료")

        # DataInterface 기반 검증용 스키마 생성
        data_interface_schema = None
        if training_df is not None:
            from src.utils.data.validation import create_data_interface_schema_for_storage
            data_interface_schema = create_data_interface_schema_for_storage(
                data_interface=self._recipe.data.data_interface,
                df=training_df,
                task_choice=self._recipe.task_choice
            )
            required_cols = len(data_interface_schema.get('required_columns', []))
            self.console.info(f"DataInterface schema 생성을 완료했습니다: {required_cols}개 필수 컬럼",
                            rich_message=f"✅ DataInterface schema 생성 완료: [cyan]{required_cols}[/cyan]개 필수 컬럼")

        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_datahandler=trained_datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            trained_calibrator=trained_calibrator,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
            data_interface_schema=data_interface_schema,
        )