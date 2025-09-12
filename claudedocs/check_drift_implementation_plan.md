# Check-Drift 기능 구현 계획

## 개요

Modern ML Pipeline에 데이터 드리프트 감지 기능을 추가한다. 학습 시점의 데이터 통계 프로필을 저장하고, 이후 새로운 데이터와 비교하여 분포 변화를 감지하는 시스템이다.

## 핵심 요구사항

### CLI 명령어
```bash
amp check-drift --run-id <mlflow_run_id> --config-path configs/dev.yaml --data-path data/new_data.csv
```

### 기능 범위
1. **학습 시 Statistical Profile 저장**: 4가지 태스크별 데이터 통계치를 MLflow에 저장
2. **드리프트 감지**: 새로운 데이터와 저장된 프로필 비교
3. **태스크별 맞춤 감지**: Regression, Classification, Timeseries, Causal Inference별 특화 로직
4. **콘솔 출력**: 변화 감지 결과와 문제 알림

## 아키텍처 설계 (Factory & Registry 패턴 활용)

### 1. 컴포넌트 구조 (기존 구조 존중)

```
src/components/drift/
├── __init__.py
├── models.py              # 데이터 구조 (Pydantic 모델)
├── registry.py            # DriftProfilerRegistry, DriftDetectorRegistry
├── profilers/             # Task별 profiler 구현체들
│   ├── __init__.py
│   ├── base_profiler.py   # BaseDriftProfiler 추상 클래스
│   ├── classification_profiler.py
│   ├── regression_profiler.py
│   ├── timeseries_profiler.py
│   └── causal_profiler.py
├── detectors/             # Task별 detector 구현체들
│   ├── __init__.py
│   ├── base_detector.py   # BaseDriftDetector 추상 클래스
│   ├── classification_detector.py
│   ├── regression_detector.py
│   ├── timeseries_detector.py
│   └── causal_detector.py
└── utils.py               # 공통 통계 계산 함수들 (PSI, K-S test 등)
```

### 2. 데이터 구조 설계

#### StatisticalProfile (Pydantic 모델)
```python
{
    "task_type": "classification",
    "data_statistics": {
        "features": {
            "numerical": {
                "age": {
                    "histogram": {"bins": [0, 10, 20, ...], "counts": [5, 15, ...]},
                    "statistics": {"mean": 35.5, "std": 12.3, "quantiles": {...}},
                    "missing_rate": 0.05
                }
            },
            "categorical": {
                "gender": {
                    "value_counts": {"M": 500, "F": 400, "Other": 10},
                    "cardinality": 3,
                    "missing_rate": 0.01
                }
            }
        },
        "target": {
            # Task별로 다른 구조
            "classification": {"class_distribution": {"0": 400, "1": 600}},
            "regression": {"histogram": {...}, "statistics": {...}},
            "timeseries": {"statistics": {...}, "temporal_properties": {...}},
            "causal": {"outcome_statistics": {...}}
        },
        "special_columns": {
            # Timeseries: timestamp 통계
            # Causal: treatment 통계
        }
    },
    "metadata": {
        "total_rows": 1000,
        "total_columns": 15,
        "collection_timestamp": "2024-01-01T00:00:00",
        "column_types": {"age": "numerical", "gender": "categorical", ...}
    }
}
```

### 3. 드리프트 감지 알고리즘

#### 수치형 컬럼
- **PSI (Population Stability Index)**: 분포 형태 비교
- **K-S Test (Kolmogorov-Smirnov)**: 통계적 유의성 검정
- **평균값 변화**: 표준편차의 ±2배 기준

#### 범주형 컬럼  
- **Chi-squared Test**: 분포 동질성 검정
- **PSI**: 카테고리별 빈도 변화
- **신규/소멸 카테고리**: 새로운 값 등장 감지

#### 태스크별 특화 감지
- **Regression/Classification**: 타겟 분포 변화
- **Timeseries**: 정상성, 자기상관성, 계절성 변화
- **Causal**: 핵심 공변량(Confounder) 변화 민감 감지

## 구현 단계

### Phase 1: Factory 통합 및 Registry 패턴

#### 1.1 Factory 확장 (`src/factory/factory.py`에 추가)
```python
def create_statistical_profiler(self) -> 'BaseDriftProfiler':
    """
    Task 타입에 따른 Statistical Profiler 생성
    settings.recipe.task_choice를 기반으로 적절한 profiler 선택
    """
    cache_key = "statistical_profiler"
    if cache_key in self._component_cache:
        return self._component_cache[cache_key]
    
    task_type = self._recipe.task_choice
    
    try:
        from src.components.drift.registry import DriftProfilerRegistry
        
        self.console.component_init(f"Statistical Profiler ({task_type})", "success")
        profiler = DriftProfilerRegistry.create(
            task_type,
            settings=self.settings
        )
        
        self._component_cache[cache_key] = profiler
        self.console.info(f"Created statistical profiler: {task_type}",
                        rich_message=f"✅ Statistical profiler: [green]{task_type}[/green]")
        return profiler
        
    except Exception as e:
        available = list(DriftProfilerRegistry.get_available_profilers().keys())
        self.console.error(f"Failed to create profiler for '{task_type}'",
                         rich_message=f"❌ Profiler creation failed: [red]{task_type}[/red]",
                         context={"available_profilers": available})
        raise

def create_drift_detector(self) -> 'BaseDriftDetector':
    """
    Task 타입에 따른 Drift Detector 생성
    settings.recipe.task_choice를 기반으로 적절한 detector 선택
    """
    cache_key = "drift_detector"
    if cache_key in self._component_cache:
        return self._component_cache[cache_key]
    
    task_type = self._recipe.task_choice
    
    try:
        from src.components.drift.registry import DriftDetectorRegistry
        
        self.console.component_init(f"Drift Detector ({task_type})", "success")
        detector = DriftDetectorRegistry.create(
            task_type,
            settings=self.settings
        )
        
        self._component_cache[cache_key] = detector
        self.console.info(f"Created drift detector: {task_type}",
                        rich_message=f"✅ Drift detector: [green]{task_type}[/green]")
        return detector
        
    except Exception as e:
        available = list(DriftDetectorRegistry.get_available_detectors().keys())
        self.console.error(f"Failed to create detector for '{task_type}'",
                         rich_message=f"❌ Detector creation failed: [red]{task_type}[/red]",
                         context={"available_detectors": available})
        raise
```

#### 1.2 데이터 구조 정의 (`src/components/drift/models.py`)
```python
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class NumericalStats(BaseModel):
    histogram: Dict[str, List[float]]  # bins, counts
    statistics: Dict[str, float]       # mean, std, quantiles
    missing_rate: float

class CategoricalStats(BaseModel):
    value_counts: Dict[str, int]
    cardinality: int
    missing_rate: float

class FeatureStats(BaseModel):
    numerical: Dict[str, NumericalStats]
    categorical: Dict[str, CategoricalStats]

class TargetStats(BaseModel):
    # Task별로 Union 타입으로 정의
    pass

class StatisticalProfile(BaseModel):
    task_type: str
    data_statistics: Dict[str, Any]
    metadata: Dict[str, Any]

class DriftDetectionResult(BaseModel):
    task_type: str
    analysis_timestamp: datetime
    feature_results: Dict[str, 'FeatureDriftResult']
    overall_drift_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    recommendations: List[str]

class FeatureDriftResult(BaseModel):
    feature_name: str
    feature_type: str  # numerical, categorical
    drift_score: float
    risk_level: str
    drift_type: str
    description: str
```

#### 1.3 Registry 패턴 구현 (`src/components/drift/registry.py`)
```python
from typing import Dict, Type, Any
from src.components.drift.profilers.base_profiler import BaseDriftProfiler
from src.components.drift.detectors.base_detector import BaseDriftDetector
from src.settings import Settings

class DriftProfilerRegistry:
    """Statistical Profiler Registry (기존 패턴과 동일)"""
    _profilers: Dict[str, Type[BaseDriftProfiler]] = {}
    
    @classmethod
    def register(cls, task_type: str, profiler_class: Type[BaseDriftProfiler]):
        """Profiler 등록"""
        cls._profilers[task_type] = profiler_class
    
    @classmethod
    def create(cls, task_type: str, settings: Settings) -> BaseDriftProfiler:
        """Task 타입에 따른 profiler 생성"""
        if task_type not in cls._profilers:
            raise ValueError(f"Unknown task type: {task_type}")
        
        profiler_class = cls._profilers[task_type]
        return profiler_class(settings=settings)
    
    @classmethod
    def get_available_profilers(cls) -> Dict[str, Type[BaseDriftProfiler]]:
        return cls._profilers.copy()

class DriftDetectorRegistry:
    """Drift Detector Registry (기존 패턴과 동일)"""
    _detectors: Dict[str, Type[BaseDriftDetector]] = {}
    
    @classmethod
    def register(cls, task_type: str, detector_class: Type[BaseDriftDetector]):
        """Detector 등록"""
        cls._detectors[task_type] = detector_class
    
    @classmethod
    def create(cls, task_type: str, settings: Settings) -> BaseDriftDetector:
        """Task 타입에 따른 detector 생성"""
        if task_type not in cls._detectors:
            raise ValueError(f"Unknown task type: {task_type}")
        
        detector_class = cls._detectors[task_type]
        return detector_class(settings=settings)
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, Type[BaseDriftDetector]]:
        return cls._detectors.copy()
```

#### 1.4 Base 클래스들 (`src/components/drift/profilers/base_profiler.py`)
```python
from abc import ABC, abstractmethod
import pandas as pd
from src.components.drift.models import StatisticalProfile
from src.settings import Settings

class BaseDriftProfiler(ABC):
    """Statistical Profiler 기본 클래스"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.task_type = settings.recipe.task_choice
        self.data_interface = settings.recipe.data.data_interface
    
    @abstractmethod
    def create_profile(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> StatisticalProfile:
        """통계 프로필 생성 (Task별 구현)"""
        pass
    
    def _analyze_numerical_features(self, data: pd.DataFrame, columns: List[str]) -> Dict:
        """공통 수치형 통계 분석 로직"""
        # 공통 구현
        pass
    
    def _analyze_categorical_features(self, data: pd.DataFrame, columns: List[str]) -> Dict:
        """공통 범주형 통계 분석 로직"""
        # 공통 구현
        pass
```

#### 1.5 Base Detector 클래스 (`src/components/drift/detectors/base_detector.py`)
```python
from abc import ABC, abstractmethod
import pandas as pd
from src.components.drift.models import StatisticalProfile, DriftDetectionResult
from src.settings import Settings

class BaseDriftDetector(ABC):
    """Drift Detector 기본 클래스"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.task_type = settings.recipe.task_choice
        self.data_interface = settings.recipe.data.data_interface
    
    @abstractmethod
    def detect_drift(
        self,
        baseline_profile: StatisticalProfile,
        current_data: pd.DataFrame,
        **kwargs
    ) -> DriftDetectionResult:
        """드리프트 감지 메인 함수 (Task별 구현)"""
        pass
    
    def _detect_numerical_drift(self, baseline_stats, current_stats) -> Dict:
        """공통 수치형 드리프트 감지 로직"""
        from src.components.drift.utils import calculate_psi, ks_test
        # 공통 구현
        pass
    
    def _detect_categorical_drift(self, baseline_stats, current_stats) -> Dict:
        """공통 범주형 드리프트 감지 로직"""
        from src.components.drift.utils import chi_square_test, detect_new_categories
        # 공통 구현
        pass
```

### Phase 2: Pipeline 구조 일관성 확보

#### 2.1 Check-Drift Pipeline 생성 (`src/pipelines/check_drift_pipeline.py`)
```python
from typing import Optional, Dict, Any
from types import SimpleNamespace

import mlflow

from src.settings import Settings
from src.factory import Factory
from src.utils.core.console_manager import RichConsoleManager
from src.utils.integrations import mlflow_integration as mlflow_utils
from src.utils.core.reproducibility import set_global_seeds

def run_check_drift_pipeline(
    settings: Settings,
    run_id: str, 
    data_path: str,
    context_params: Optional[Dict[str, Any]] = None
):
    """
    데이터 드리프트 감지 파이프라인을 실행합니다.
    저장된 statistical profile과 새로운 데이터를 비교하여 드리프트를 감지합니다.
    """
    console = RichConsoleManager()
    
    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)
    
    # Pipeline context start
    pipeline_description = f"Baseline Run ID: {run_id} | Environment: {settings.config.environment.name}"
    
    with console.pipeline_context("Drift Detection Pipeline", pipeline_description):
        context_params = context_params or {}
        
        # MLflow 실행 컨텍스트 시작 (새로운 run 생성)
        run_name = f"drift_detection_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow_utils.start_run(settings, run_name=run_name) as drift_run:
            drift_run_id = drift_run.info.run_id
            
            # Factory 생성
            factory = Factory(settings)
            
            # 1. Statistical Profile 다운로드
            console.log_phase("Profile Loading", "📊")
            baseline_profile = _load_statistical_profile(run_id, console)
            
            # 2. 새로운 데이터 로드
            console.log_phase("Data Loading", "📥")
            current_data = _load_drift_data(factory, data_path, context_params, console)
            
            # 3. Drift Detection 컴포넌트 생성
            console.log_phase("Component Initialization", "🔧")
            drift_detector = factory.create_drift_detector()
            
            # 4. 드리프트 감지 실행
            console.log_phase("Drift Detection", "🔍")
            drift_result = drift_detector.detect_drift(
                baseline_profile=baseline_profile,
                current_data=current_data
            )
            
            # 5. 결과 출력 및 저장
            console.log_phase("Results & Reporting", "📋")
            _display_drift_results(drift_result, console)
            _log_drift_results(drift_result)
            
            return SimpleNamespace(
                drift_run_id=drift_run_id,
                baseline_run_id=run_id,
                overall_risk_level=drift_result.risk_level,
                drift_score=drift_result.overall_drift_score
            )
```

### Phase 3: Train Pipeline 통합

#### 3.1 Train Pipeline 수정 (`src/pipelines/train_pipeline.py`)
```python
# 기존 코드에 추가
from src.components.drift.statistical_profiler import StatisticalProfiler

def run_train_pipeline(settings, context_params=None, record_requirements=False):
    # ... 기존 코드 ...
    
    # 4. 데이터 준비 (기존)
    X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data = datahandler.split_and_prepare(augmented_df)
    
    # **NEW: Statistical Profile 생성 및 저장**
    console.log_phase("Statistical Profiling", "📊")
    profiler = StatisticalProfiler(task_type=settings.recipe.task_choice)
    
    # 학습 데이터 기준으로 프로필 생성
    statistical_profile = profiler.create_profile(
        data=augmented_df,  # 원본 데이터
        target_col=settings.recipe.data.data_interface.target_column,
        timestamp_col=getattr(settings.recipe.data.data_interface, 'timestamp_column', None),
        treatment_col=getattr(settings.recipe.data.data_interface, 'treatment_column', None)
    )
    
    # MLflow에 아티팩트로 저장
    import json
    profile_path = "statistical_profile.json"
    with open(profile_path, 'w') as f:
        json.dump(statistical_profile.model_dump(), f, indent=2)
    mlflow.log_artifact(profile_path, "drift_detection")
    
    console.log_milestone("Statistical profile saved to MLflow", "success")
    
    # ... 나머지 기존 코드 계속 ...
```

### Phase 4: CLI 명령어 단순화 (기존 패턴 준수)

#### 4.1 Check-Drift 명령어 (`src/cli/commands/check_drift_command.py`) - 단순화
```python
import typer
import mlflow
import json
import pandas as pd
from typing_extensions import Annotated
from pathlib import Path

from src.settings import load_settings
from src.components.drift.drift_detector import DriftDetector
from src.components.drift.models import StatisticalProfile
from src.utils.core.logger import setup_logging, logger

def check_drift_command(
    run_id: Annotated[str, typer.Option("--run-id", help="MLflow run ID")],
    config_path: Annotated[str, typer.Option("--config-path", "-c", help="Config 파일 경로")],
    data_path: Annotated[str, typer.Option("--data-path", "-d", help="비교할 새 데이터 파일 경로")]
) -> None:
    """
    데이터 드리프트 감지 실행
    
    저장된 statistical profile과 새로운 데이터를 비교하여
    분포 변화를 감지하고 결과를 출력합니다.
    """
    try:
            # 1. 설정 로드 (기존 패턴과 동일)
        config_data = load_config_files(config_path=config_path)
        settings = create_settings_for_inference(config_data)  # inference와 동일한 패턴
        setup_logging(settings)
        
        # 2. 드리프트 감지 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Baseline Run ID: {run_id}")
        
        # 3. 드리프트 감지 파이프라인 실행 (inference와 동일한 구조)
        run_check_drift_pipeline(
            settings=settings,
            run_id=run_id,
            data_path=data_path,
            context_params=params or {},
        )
        
        logger.info("✅ 드리프트 감지가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"드리프트 감지 중 오류 발생: {e}", exc_info=True)
        raise typer.Exit(code=1)

def _display_drift_results(result):
    """드리프트 감지 결과를 콘솔에 출력"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # 전체 요약
    console.print(f"\n[bold blue]🔍 Data Drift Detection Results[/bold blue]")
    console.print(f"Task Type: {result.task_type}")
    console.print(f"Analysis Time: {result.analysis_timestamp}")
    
    # 위험 수준별 요약
    high_risk_features = [f for f, r in result.feature_results.items() if r.risk_level == "HIGH"]
    medium_risk_features = [f for f, r in result.feature_results.items() if r.risk_level == "MEDIUM"]
    
    if high_risk_features:
        console.print(f"\n[bold red]🚨 HIGH RISK FEATURES ({len(high_risk_features)})[/bold red]")
        for feature in high_risk_features:
            details = result.feature_results[feature]
            console.print(f"  • {feature}: {details.drift_type} (Score: {details.drift_score:.3f})")
    
    if medium_risk_features:
        console.print(f"\n[bold yellow]⚠️ MEDIUM RISK FEATURES ({len(medium_risk_features)})[/bold yellow]")
        for feature in medium_risk_features:
            details = result.feature_results[feature]
            console.print(f"  • {feature}: {details.drift_type} (Score: {details.drift_score:.3f})")
    
    # 상세 테이블
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Feature", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Drift Score", justify="right")
    table.add_column("Risk Level", justify="center")
    table.add_column("Details")
    
    for feature, details in result.feature_results.items():
        risk_color = {
            "HIGH": "red",
            "MEDIUM": "yellow", 
            "LOW": "green"
        }.get(details.risk_level, "white")
        
        table.add_row(
            feature,
            details.feature_type,
            f"{details.drift_score:.3f}",
            f"[{risk_color}]{details.risk_level}[/{risk_color}]",
            details.description
        )
    
    console.print(table)
    
    # 권장 사항
    if high_risk_features or medium_risk_features:
        console.print(f"\n[bold yellow]📋 Recommendations:[/bold yellow]")
        if high_risk_features:
            console.print("• Consider retraining the model due to significant data drift")
            console.print("• Investigate root causes of distribution changes")
        if medium_risk_features:
            console.print("• Monitor these features closely in future predictions")
            console.print("• Consider updating feature engineering pipeline")
```

#### 4.2 메인 CLI에 명령어 등록 (`src/cli/main_commands.py`)
```python
# 기존 import에 추가
from src.cli.commands.check_drift_command import check_drift_command

# ML Pipeline Commands 섹션에 추가 (train, batch-inference, serve-api와 동일한 위치)
app.command("check-drift", help="데이터 드리프트 감지 실행")(check_drift_command)
```

## 태스크별 드리프트 감지 세부 사항

### 1. Regression & Classification
- **입력 피처**: PSI > 0.25 또는 K-S test p < 0.05
- **타겟 변수**: 
  - Regression: 분포 변화 감지
  - Classification: 클래스 불균형 변화 감지

### 2. Timeseries  
- **기본 피처**: Regression과 동일
- **시간 속성**: 
  - 정상성 변화 (ADF test)
  - 자기상관 패턴 변화 (ACF)
  - 계절성 강도 변화

### 3. Causal Inference
- **핵심 공변량**: 다른 태스크보다 더 엄격한 기준 (PSI > 0.2)
- **처치/결과 변수**: 분포 변화 감지
- **인과관계 왜곡**: 공변량 변화가 처치효과에 미치는 영향 평가

### 4. Clustering
- **입력 피처만**: 타겟이 없으므로 피처 드리프트에만 집중
- **구조적 변화**: 데이터 밀도나 클러스터 형성에 영향을 줄 수 있는 변화 감지

## 성능 고려사항

### 1. 대용량 데이터 처리
- **샘플링**: 10만 행 이상 시 stratified sampling
- **배치 처리**: 메모리 효율적인 통계 계산
- **병렬 처리**: 컬럼별 병렬 분석

### 2. 계산 최적화
- **히스토그램**: 고정된 bin 수 (50개) 사용
- **PSI 계산**: NumPy vectorization 활용
- **통계 검정**: scipy.stats 최적화된 함수 사용

## 에러 핸들링

### 1. 데이터 호환성
- **스키마 불일치**: 컬럼명/타입 변경 감지
- **결측치 처리**: 새로운 결측 패턴 감지
- **데이터 타입**: 자동 형변환 및 경고

### 2. MLflow 연동
- **아티팩트 누락**: 명확한 에러 메시지
- **권한 오류**: 접근 권한 확인
- **네트워크 오류**: 재시도 로직

## 테스트 전략

### 1. 단위 테스트
- 각 통계 계산 함수
- 드리프트 감지 알고리즘
- 데이터 타입 감지

### 2. 통합 테스트  
- Train → Check-drift 전체 플로우
- 다양한 데이터 형식 지원
- MLflow 연동 테스트

### 3. 성능 테스트
- 대용량 데이터 처리 시간
- 메모리 사용량 측정
- 정확도 벤치마크

## 향후 확장성

### 1. 고급 드리프트 감지
- **Multivariate drift**: 피처 간 상관관계 변화
- **Concept drift**: 입출력 관계 변화
- **Temporal drift**: 시간에 따른 점진적 변화

### 2. 자동화 기능
- **임계값 학습**: 도메인별 최적 임계값 자동 설정
- **알림 시스템**: Slack/Email 자동 알림
- **재학습 트리거**: 심각한 드리프트 시 자동 재학습

### 3. 시각화
- **드리프트 대시보드**: 웹 기반 모니터링
- **히스토그램 비교**: 분포 변화 시각화
- **시계열 추적**: 드리프트 점수 추이

## 결론

이 구현을 통해 Modern ML Pipeline에 강력한 데이터 드리프트 감지 기능이 추가된다. 
- **학습 시**: 자동으로 statistical profile 저장
- **운영 시**: 간단한 CLI 명령어로 드리프트 감지
- **태스크별**: 각 ML 태스크에 특화된 감지 로직
- **확장성**: 향후 고급 기능 추가 가능한 모듈 구조