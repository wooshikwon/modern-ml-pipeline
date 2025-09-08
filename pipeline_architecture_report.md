# Modern ML Pipeline 아키텍처 분석 보고서

## 📋 프로젝트 개요

**Modern ML Pipeline (MMP)**는 YAML 기반 설정 주도 머신러닝 파이프라인 프레임워크입니다. CLI 명령에서 시작하여 Pipeline을 거쳐 Factory 패턴으로 컴포넌트를 생성하는 체계적인 아키텍처를 구현하고 있습니다.

### 핵심 특징
- **설정 주도 아키텍처**: Recipe(모델 정의) + Config(환경 설정) YAML 파일 기반
- **통합 CLI 인터페이스**: `mmp` 명령으로 학습/추론/서빙 통합 관리
- **Factory 패턴**: 중앙화된 컴포넌트 생성 및 의존성 관리
- **MLflow 통합**: 실험 추적, 모델 저장, 버전 관리

## 🏗️ 아키텍처 구조

### 1. CLI 진입점 계층

```
src/__main__.py (Entry Point)
    └── src/cli/main_commands.py (Router)
            ├── train_command.py      → run_train_pipeline()
            ├── inference_command.py  → run_inference_pipeline()  
            └── serve_command.py       → run_api_server()
```

#### CLI 명령 체계
- **`mmp train`**: 모델 학습 파이프라인 실행
- **`mmp batch-inference`**: 배치 추론 실행
- **`mmp serve-api`**: REST API 서버 실행
- **`mmp init`**: 프로젝트 초기화
- **`mmp system-check`**: 시스템 연결 상태 검사

### 2. Pipeline 계층

#### 2.1 Train Pipeline (`src/pipelines/train_pipeline.py`)

**실행 흐름:**
```python
run_train_pipeline(settings, context_params)
    ├── 1. Factory 생성: Factory(settings)
    ├── 2. 데이터 로딩: factory.create_data_adapter() → adapter.read()
    ├── 3. 컴포넌트 생성:
    │      ├── factory.create_fetcher()
    │      ├── factory.create_datahandler()
    │      ├── factory.create_preprocessor()
    │      ├── factory.create_model()
    │      ├── factory.create_evaluator()
    │      └── factory.create_trainer()
    ├── 4. 학습 실행: trainer.train(df, model, ...)
    ├── 5. PyfuncWrapper 생성: factory.create_pyfunc_wrapper()
    └── 6. MLflow 저장: mlflow.pyfunc.log_model()
```

**핵심 포인트:**
- Factory를 통한 모든 컴포넌트의 중앙화된 생성
- PyfuncWrapper로 모델과 전처리기를 캡슐화
- MLflow에 모델과 메타데이터 저장

#### 2.2 Inference Pipeline (`src/pipelines/inference_pipeline.py`)

**실행 흐름:**
```python
run_inference_pipeline(settings, run_id, data_path, context_params)
    ├── 1. MLflow 모델 로드: mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    ├── 2. Factory 생성: Factory(settings)
    ├── 3. 데이터 로딩:
    │      ├── CLI data_path 우선 사용
    │      ├── Jinja 템플릿 렌더링 지원 (.sql.j2)
    │      └── factory.create_data_adapter() → adapter.read()
    ├── 4. 예측 실행: model.predict(df)
    ├── 5. 메타데이터 추가: model_run_id, inference_run_id, timestamp
    └── 6. 결과 저장: Storage/SQL/BigQuery 어댑터로 저장
```

**핵심 포인트:**
- 학습된 모델 재사용 (MLflow Run ID 기반)
- 동적 SQL 템플릿 지원 (Jinja2)
- 다양한 출력 어댑터 지원

#### 2.3 Serving Pipeline (`src/serving/router.py`)

**실행 흐름:**
```python
run_api_server(settings, run_id, host, port)
    ├── 1. FastAPI 앱 생성
    ├── 2. MLflow 모델 로드 (lifespan 이벤트)
    ├── 3. 엔드포인트 등록:
    │      ├── GET /health → 헬스 체크
    │      ├── POST /predict → 단일 예측
    │      ├── GET /model/metadata → 모델 메타데이터
    │      └── GET /model/optimization → 최적화 히스토리
    └── 4. Uvicorn 서버 실행
```

**핵심 포인트:**
- FastAPI 기반 REST API
- 모델 자기 기술 엔드포인트
- 실시간 예측 서비스

### 3. Factory 계층 (`src/factory/factory.py`)

#### Factory 클래스 구조

```python
class Factory:
    def __init__(self, settings: Settings):
        self._ensure_components_registered()  # 컴포넌트 레지스트리 초기화
        self._component_cache = {}            # 생성된 컴포넌트 캐싱
    
    # 핵심 생성 메서드들
    def create_data_adapter(adapter_type=None) → BaseAdapter
    def create_fetcher(run_mode=None) → BaseFetcher
    def create_preprocessor() → BasePreprocessor
    def create_model() → Any
    def create_evaluator() → BaseEvaluator
    def create_trainer() → BaseTrainer
    def create_datahandler() → BaseDataHandler
    def create_pyfunc_wrapper(...) → PyfuncWrapper
```

#### Factory 패턴의 장점

1. **중앙화된 의존성 관리**: 모든 컴포넌트 생성이 한 곳에서 관리
2. **캐싱 메커니즘**: 동일 컴포넌트 재사용으로 성능 최적화
3. **동적 컴포넌트 생성**: class_path 기반 런타임 객체 생성
4. **일관된 생성 패턴**: Registry 패턴과 결합된 표준화된 생성 로직

#### 컴포넌트 레지스트리 시스템

```python
# 각 컴포넌트는 자체 Registry를 가짐
AdapterRegistry.create("sql", settings)
FetcherRegistry.create("feature_store", settings)
EvaluatorRegistry.create("classification", settings)
TrainerRegistry.create("default", settings)
DataHandlerRegistry.get_handler_for_task("classification", settings)
```

### 4. 인터페이스 추상화 계층 (`src/interface/`)

#### 핵심 추상 클래스들

1. **BaseAdapter**: 데이터 읽기/쓰기 표준 인터페이스
   ```python
   class BaseAdapter(ABC):
       @abstractmethod
       def read(source: str) → pd.DataFrame
       @abstractmethod
       def write(df: pd.DataFrame, target: str)
   ```

2. **BaseTrainer**: 학습 파이프라인 표준 인터페이스
   ```python
   class BaseTrainer(ABC):
       @abstractmethod
       def train(df: pd.DataFrame) → Tuple[model, preprocessor, metrics]
   ```

3. **BaseFactory**: 팩토리 표준 인터페이스
   ```python
   class BaseFactory(ABC):
       @abstractmethod
       def create_model()
       @abstractmethod
       def create_pyfunc_wrapper(model, preprocessor) → mlflow.pyfunc.PythonModel
   ```

## 🔄 데이터 흐름 분석

### Train 워크플로우
```
CLI 명령 입력
    ↓ (recipe + config + data_path)
Settings 생성
    ↓
Factory 초기화
    ↓
데이터 어댑터 생성 → 데이터 로딩
    ↓
컴포넌트 생성 (fetcher, preprocessor, model, evaluator, trainer)
    ↓
Trainer.train() 실행
    ↓
PyfuncWrapper 생성 (모델 + 전처리기 캡슐화)
    ↓
MLflow 저장 (모델 + 메타데이터 + 스키마)
```

### Inference 워크플로우
```
CLI 명령 입력
    ↓ (run_id + config + data_path)
MLflow 모델 로드
    ↓
Factory 초기화
    ↓
데이터 어댑터 생성 → 데이터 로딩
    ↓ (Jinja 템플릿 렌더링 지원)
모델 예측 실행
    ↓
메타데이터 추가 (run_id, timestamp)
    ↓
결과 저장 (Storage/SQL/BigQuery)
```

### Serving 워크플로우
```
CLI 명령 입력
    ↓ (run_id + config + host:port)
FastAPI 앱 생성
    ↓
MLflow 모델 로드 (startup)
    ↓
REST API 엔드포인트 활성화
    ↓
실시간 예측 요청 처리
```

## 💡 핵심 설계 패턴

### 1. Factory 패턴
- **목적**: 객체 생성 로직의 캡슐화와 중앙화
- **구현**: `Factory` 클래스가 모든 컴포넌트 생성 담당
- **이점**: 의존성 관리 단순화, 테스트 용이성, 확장성

### 2. Registry 패턴
- **목적**: 컴포넌트의 동적 등록과 검색
- **구현**: 각 컴포넌트 타입별 Registry 클래스
- **이점**: 플러그인 아키텍처, 런타임 컴포넌트 추가

### 3. Adapter 패턴
- **목적**: 다양한 데이터 소스와의 통합 인터페이스
- **구현**: `BaseAdapter` 추상 클래스와 구체 구현체들
- **이점**: 데이터 소스 독립성, 확장 가능한 I/O

### 4. Template Method 패턴
- **목적**: 알고리즘 골격 정의, 세부 구현은 서브클래스에 위임
- **구현**: `BaseTrainer`, `BaseEvaluator` 등의 추상 클래스
- **이점**: 코드 재사용, 일관된 처리 흐름

### 5. Strategy 패턴
- **목적**: 런타임에 알고리즘 선택
- **구현**: task_choice에 따른 다른 DataHandler/Evaluator 선택
- **이점**: 유연한 태스크 처리, 확장 가능한 알고리즘

## 🎯 주요 특징

### 1. 설정 주도 아키텍처
- **Recipe YAML**: 모델 정의, 하이퍼파라미터, 데이터 인터페이스
- **Config YAML**: 환경별 설정 (개발/운영), 연결 정보
- **동적 조합**: 런타임에 Recipe + Config 조합

### 2. 캐싱 메커니즘
- **컴포넌트 캐싱**: Factory 내부에서 생성된 컴포넌트 재사용
- **성능 최적화**: 중복 생성 방지, 메모리 효율성

### 3. 데이터 검증
- **DataInterface**: 스키마 기반 컬럼 검증
- **타입 체크**: 학습/추론 시 데이터 타입 일관성 보장
- **필수 컬럼 검증**: entity, target, timestamp 컬럼 확인

### 4. MLflow 통합
- **실험 추적**: 모든 학습 실행 자동 기록
- **모델 버전 관리**: Run ID 기반 모델 관리
- **메타데이터 저장**: 하이퍼파라미터, 메트릭, 스키마

### 5. 확장 가능한 아키텍처
- **플러그인 시스템**: Registry 패턴으로 새 컴포넌트 추가 용이
- **다양한 데이터 소스**: SQL, BigQuery, Storage, Feature Store
- **모델 독립성**: sklearn, xgboost, lightgbm, custom 모델 지원

## 📊 컴포넌트 관계도

```
Settings (Recipe + Config)
    ↓
Factory (중앙 생성자)
    ├── DataAdapter (SQL/Storage/BigQuery)
    ├── Fetcher (PassThrough/FeatureStore)
    ├── DataHandler (Classification/Regression/Uplift)
    ├── Preprocessor (전처리 파이프라인)
    ├── Model (ML 모델)
    ├── Evaluator (평가 메트릭)
    ├── Trainer (학습 오케스트레이터)
    └── PyfuncWrapper (MLflow 래퍼)
```

## 🚀 강점과 개선 가능 영역

### 강점
1. **명확한 관심사 분리**: CLI → Pipeline → Factory → Component
2. **높은 확장성**: Registry와 Factory 패턴으로 새 컴포넌트 추가 용이
3. **설정 주도**: 코드 변경 없이 YAML로 동작 변경
4. **포괄적인 추상화**: 인터페이스로 구현 세부사항 은닉
5. **MLflow 통합**: 완벽한 실험 추적과 모델 관리

### 개선 가능 영역
1. **의존성 주입**: Factory에서 더 명시적인 DI 패턴 활용 가능
2. **비동기 처리**: 대용량 데이터 처리를 위한 async/await 지원
3. **에러 처리**: 더 세분화된 예외 계층 구조
4. **테스트 커버리지**: 통합 테스트 강화 필요
5. **문서화**: API 문서와 사용 예제 확충

## 🎭 결론

Modern ML Pipeline은 **CLI → Pipeline → Factory** 흐름을 통해 명확하고 확장 가능한 아키텍처를 구현한 우수한 ML 프레임워크입니다. Factory 패턴을 중심으로 한 컴포넌트 생성 전략과 Registry 패턴을 통한 동적 등록 시스템은 프로젝트의 유지보수성과 확장성을 크게 향상시킵니다. 

특히 설정 주도 접근법과 MLflow 통합은 실제 프로덕션 환경에서의 ML 워크플로우 관리를 효과적으로 지원하며, 다양한 데이터 소스와 모델 타입을 유연하게 처리할 수 있는 구조를 제공합니다.