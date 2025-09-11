# Practical Tests Structure Refactoring Strategy

## Executive Summary

**현재 상태**: 62/62 integration tests 100% 성공  
**핵심 문제**: 테스트 코드의 70-80%가 setup, 20-30%만 실제 검증  
**해결 방안**: **Context-based Test Architecture** - 책임 분리와 setup 중앙화  
**마이그레이션**: **Zero-risk incremental migration** - 100% 성공률 보장

---

## ⚡ Quickstart: 새 Context 기반 테스트 작성

1. 컨텍스트 픽스처 사용: `mlflow_test_context` 또는 목적에 맞는 컨텍스트 픽스처를 테스트 시그니처에 추가
2. 컨텍스트 매니저로 설정/데이터/리소스 자동 준비: `with ... as ctx:`
3. 파이프라인/퍼블릭 API 호출 후, 컨텍스트의 검증 헬퍼로 관찰/검증만 수행

```python
def test_quickstart_mlflow(mlflow_test_context):
    with mlflow_test_context.for_classification(experiment="quickstart") as ctx:
        result = run_train_pipeline(ctx.settings)
        assert result is not None
        assert ctx.experiment_exists()
        assert ctx.get_experiment_run_count() == 1
        assert isinstance(ctx.get_run_metrics(), dict)
```

## 🔤 Naming Conventions

- **Context 파일**: `tests/fixtures/contexts/<area>_context.py` (예: `mlflow_context.py`)
- **Context 클래스**: `*TestContext`, 내부 매니저는 `*ContextManager`
- **픽스처 이름**: `<area>_test_context` (예: `mlflow_test_context`)
- **테스트 함수**: `test_<area>_<behavior>[_v2]` (A/B 공존 시 `_v2` 접미사)
- **템플릿 파일**: `tests/fixtures/templates/configs/<task>_base.yaml`

## 🚫 Testing Anti-Patterns

- **비즈니스 로직 재구현**: 컨텍스트 내부에서 파이프라인/팩토리 로직을 재현하지 않는다(퍼블릭 API만 호출).
- **상태 공유**: 테스트 간 `Factory`/`Settings`/리소스를 공유하지 않는다(테스트당 새 생성).
- **시간 기반 이름**: `time.time()` 등 시간 의존적 명명 금지 → uuid 기반 사용.
- **숨은 전역 변경**: 글로벌 환경 변수/작업 디렉토리 변경은 컨텍스트 범위 내에서만 하고 자동 복원.
- **불필요한 모킹**: 레지스트리/팩토리 경로를 모킹하지 말고, 실제 경량 컴포넌트로 검증.

## 🧰 Operational Considerations

- **데이터 수명주기**: 모든 임시 파일은 테스트 전용 temp 디렉토리 안에서 생성/삭제.
- **MLflow 저장소**: `file://{temp_dir}/mlruns` 고정 사용(외부 경로 불가).
- **격리**: 테스트당 1 run 기준, 교차 테스트 의존 금지.
- **성능 예산**: 컨텍스트 초기화/파이프라인 실행에 대해 상한선 설정 및 측정(`performance_benchmark`).
- **요구사항 캡처(선택)**: 기본적으로 MLflow 아티팩트에 패키지 요구사항을 기록하지 않음. 필요 시 `mmp train --record-reqs`로 활성화. (이전 환경변수 가드 `MLPIPE_SKIP_PIP_REQ`는 제거)
 - **데이터 경로 주입 정책**: 데이터 경로는 CLI `--data-path`(또는 테스트 컨텍스트 빌더의 `with_data_path`)로만 주입한다. 레시피에는 `loader.source_uri`를 저장하지 않는다.

### ⏱️ Timeseries 규약 및 모델/카탈로그 매칭

- **Timeseries 규약(필수)**
  - `recipe.data.data_interface.timestamp_column`은 Timeseries 작업에서 반드시 지정해야 한다. 누락 시 Validator가 실패하도록 강제한다.
  - Feature Store 사용 시 `data.fetcher.timestamp_column`을 함께 지정하는 것을 권장한다(Point-in-Time join 기준 컬럼).
- **카탈로그-클래스 매칭 규칙**
  - 모델 카탈로그의 `class_path`는 런타임에 그대로 임포트 가능한 경로여야 한다.
  - DataHandler 선택은 카탈로그의 `data_handler`가 우선한다. Timeseries 작업이라도 LSTM처럼 시퀀스 전처리가 필요한 경우 `data_handler: deeplearning`이 올바른 설정이다.
  - 예: `src/models/catalog/Timeseries/LSTMTimeSeries.yaml`은 `task_type: timeseries`이면서 `data_handler: deeplearning`을 사용한다(정상).

### 🧭 Feature Store 가이드

- Feature Store(fetcher `type: feature_store`)를 사용할 때는 다음을 권장/요구한다:
  - `data.fetcher.timestamp_column`을 반드시 지정한다(권장 수준, 운영에서는 사실상 필수).
  - 각 `feature_views.*.join_key`와 `features`를 명시하고, 테스트에서는 간단한 뷰 1-2개만 사용하여 빠르게 검증한다.

---

## 🎯 Core Problems: 현재 구조의 진짜 문제점

### Problem 1: Setup Overhead Dominance

**현재 테스트 구조 분석**:
```python
def test_mlflow_experiment_creation_and_tracking(self, isolated_temp_directory, settings_builder):
    # ━━━ SETUP CODE (23 lines, 80% of test) ━━━
    mlflow_uri = f"sqlite:///{isolated_temp_directory}/test_mlflow.db"
    experiment_name = f"integration_test_{int(time.time())}"
    
    test_data = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'target': np.random.randint(0, 2, 50)
    })
    data_path = isolated_temp_directory / "mlflow_test_data.csv"
    test_data.to_csv(data_path, index=False)
    
    settings = settings_builder \
        .with_task("classification") \
        .with_model("sklearn.ensemble.RandomForestClassifier") \
        .with_data_path(str(data_path)) \
        .with_mlflow(tracking_uri=mlflow_uri, experiment_name=experiment_name) \
        .build()
    
    # ━━━ ACTUAL TEST LOGIC (3 lines, 20% of test) ━━━
    try:
        result = run_train_pipeline(settings)
        assert result is not None
    except Exception as e:
        assert True  # No Mock Hell validation
```

**문제**: 테스트의 진짜 목적인 "MLflow experiment creation validation"이 코드의 20%만 차지

### Problem 2: Responsibility Mixing

**한 테스트가 담당하는 4가지 책임**:
1. **Data Generation**: Test data 생성 및 파일 저장
2. **Resource Setup**: MLflow/Database URI 생성  
3. **Settings Configuration**: Settings 객체 구성
4. **Business Logic Validation**: 실제 테스트하려는 비즈니스 로직 (진짜 목적)

**결과**: 테스트 의도 불분명, 코드 중복, 유지보수 어려움

### Problem 3: Template Duplication

**62개 테스트에서 반복되는 패턴들**:

**Pattern A: MLflow Setup (11회 반복)**
```python
mlflow_uri = f"sqlite:///{isolated_temp_directory}/xxx.db"
experiment_name = f"yyy_{int(time.time())}"
```

**Pattern B: YAML Configuration (15회 반복)**  
```python
config_yaml = """
environment:
  name: integration_test
data_source:
  adapter_type: storage
mlflow:
  tracking_uri: sqlite:///zzz.db
"""
```

**Pattern C: Component Factory Setup (30회 반복)**
```python
factory = Factory(settings)
adapter = factory.create_data_adapter()
model = factory.create_model()
```

---

## 🏗️ Proposed Architecture: Context-based Test Structure

### Core Design Principles

1. **Separation of Concerns**: Setup vs Verification 책임 분리
2. **Context Management**: 각 테스트 영역별 전용 Context 클래스
3. **Zero Configuration**: 최소한의 설정으로 최대한의 setup
4. **Incremental Migration**: 기존 테스트와 새 구조 공존

#### Context Minimal Contract

- **역할 최소화**: 컨텍스트는 오케스트레이션을 "호출"하고 결과를 "관찰"만 한다.
- **퍼블릭 API만 사용**: `run_train_pipeline`, `Factory.create_*` 등 공개 API만 호출하고 비즈니스 로직 재구현은 금지한다.
- **필수 속성**: `ctx.settings`, `ctx.data_path`, `ctx.tracking_uri`, `ctx.experiment_name`
- **필수 헬퍼**: `experiment_exists()`, `get_experiment_run_count()`, `get_run_metrics()`
- **선택 헬퍼(MLflow 확장)**: `verify_mlflow_artifacts()`
- **상태 격리**: 테스트마다 새 `Settings`/새 `Factory` 생성(컴포넌트 캐시/상태 누수 방지)

### Architecture Overview

```
tests/
├── conftest.py                     # 기존 + 새로운 context fixtures
├── fixtures/
│   ├── contexts/                   # 🆕 Context Classes
│   │   ├── __init__.py
│   │   ├── mlflow_context.py      # MLflow Test Context
│   │   ├── database_context.py    # Database Test Context
│   │   ├── component_context.py   # Component Factory Context
│   │   └── scenario_context.py    # End-to-end Scenario Context
│   ├── templates/                  # 🆕 YAML Templates
│   │   ├── configs/
│   │   │   ├── classification_base.yaml
│   │   │   ├── regression_base.yaml
│   │   │   └── mlflow_base.yaml
│   │   └── scenarios/
│   │       ├── classification_full.yaml
│   │       └── regression_full.yaml
│   ├── expected/                   # 🆕 Enhanced Expected Outputs
│   │   ├── metrics/
│   │   │   ├── classification_baseline.json
│   │   │   └── regression_baseline.json
│   │   ├── predictions/
│   │   │   ├── sample_classification.csv
│   │   │   └── sample_regression.csv
│   │   └── responses/
│   │       ├── mlflow_tracking.json
│   │       └── component_interactions.json
│   └── [기존 디렉토리들 유지]
└── integration/                    # 기존 테스트 + 새 버전 공존
```

### Standardization for Tests

- **MLflow tracking_uri**: `file://{temp_dir}/mlruns` (테스트 전용, 로컬 경로 고정)
- **Experiment name**: `{prefix}-{uuid4().hex[:8]}` (시간 의존 제거, 충돌 방지)
- **데이터 생성 시드**: 모든 컨텍스트에서 고정 시드 사용(`seed=42` 기본)
- **상태 격리 원칙**: 각 테스트에서 `Settings`/`Factory`를 새로 생성

---

## 🔧 Implementation Examples

### Example 1: MLflow Test Context

**Before (현재 방식)**:
```python
def test_mlflow_experiment_creation(self, isolated_temp_directory, settings_builder):
    # 23 lines of setup code
    mlflow_uri = f"sqlite:///{isolated_temp_directory}/test_mlflow.db"
    experiment_name = f"integration_test_{int(time.time())}"
    test_data = pd.DataFrame({...})
    data_path = isolated_temp_directory / "test.csv" 
    test_data.to_csv(data_path, index=False)
    settings = settings_builder.with_task("classification")...build()
    
    # 3 lines of actual test
    try:
        result = run_train_pipeline(settings)
        assert result is not None
    except Exception:
        assert True
```

**After (Context 방식)**:
```python
def test_mlflow_experiment_creation(self, mlflow_test_context):
    with mlflow_test_context.for_classification(experiment="experiment_creation") as ctx:
        result = run_train_pipeline(ctx.settings)
        assert result is not None
        assert ctx.experiment_exists()
        assert ctx.get_experiment_run_count() == 1
        metrics = ctx.get_run_metrics()
        assert isinstance(metrics, dict) and len(metrics) > 0
```

**MLflowTestContext 구현**:
```python
from uuid import uuid4
import pandas as pd
from mlflow.tracking import MlflowClient

class MLflowTestContext:
    def __init__(self, isolated_temp_directory, settings_builder, test_data_generator, seed: int = 42):
        self.temp_dir = isolated_temp_directory
        self.settings_builder = settings_builder
        self.data_generator = test_data_generator
        self.seed = seed
        
    def for_classification(self, experiment: str, model: str = "RandomForestClassifier"):
        return MLflowContextManager(
            task="classification",
            experiment_suffix=experiment,
            model_class=f"sklearn.ensemble.{model}",
            context=self
        )

class MLflowContextManager:
    def __enter__(self):
        # 1) MLflow URI 표준화
        self.mlflow_uri = f"file://{self.context.temp_dir}/mlruns"
        # 2) 실험명은 uuid 기반
        self.experiment_name = f"{self.experiment_suffix}-{uuid4().hex[:8]}"
        
        # 3) 결정론적 데이터 생성
        X, y = self.context.data_generator.classification_data(n_samples=50, n_features=4, random_state=self.context.seed)
        self.test_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        self.test_data['target'] = y
        
        self.data_path = self.context.temp_dir / f"data_{self.experiment_suffix}.csv"
        self.test_data.to_csv(self.data_path, index=False)
        
        # 4) Settings 자동 구성
        self.settings = self.context.settings_builder \
            .with_task(self.task) \
            .with_model(self.model_class) \
            .with_data_path(str(self.data_path)) \
            .with_mlflow(self.mlflow_uri, self.experiment_name) \
            .build()
        
        # 5) MLflow client 준비 및 experiment id 확보
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        exp = self.mlflow_client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            self.experiment_id = self.mlflow_client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = exp.experiment_id
        
        return self
        
    def experiment_exists(self):
        return self.mlflow_client.get_experiment_by_name(self.experiment_name) is not None
    
    def get_experiment_run_count(self):
        runs = self.mlflow_client.search_runs([self.experiment_id])
        return len(runs)
    
    def get_run_metrics(self):
        runs = self.mlflow_client.search_runs([self.experiment_id], max_results=1, order_by=["attributes.start_time DESC"])
        if not runs:
            return {}
        return runs[0].data.metrics
```

### Example 2: Component Test Context

**Before**:
```python
def test_adapter_to_model_data_flow(self, isolated_temp_directory, settings_builder, test_data_generator):
    # 15 lines of component setup
    X, y = test_data_generator.classification_data(50, 4)
    test_data = pd.DataFrame(X, columns=[...])
    data_path = isolated_temp_directory / "dataflow_test.csv"
    test_data.to_csv(data_path, index=False)
    settings = settings_builder.with_task("classification")...build()
    factory = Factory(settings)
    adapter = factory.create_data_adapter()
    model = factory.create_model()
    
    # 5 lines of actual test
    raw_data = adapter.read(str(data_path))
    X_train = raw_data[feature_columns]
    y_train = raw_data[target_column]
    assert len(X_train) > 0
    assert len(y_train) > 0
```

**After**:
```python
def test_adapter_to_model_data_flow(self, component_test_context):
    with component_test_context.classification_stack() as ctx:
        # 8 lines of focused data flow verification
        raw_data = ctx.adapter.read(ctx.data_path)
        processed_data = ctx.prepare_model_input(raw_data)
        
        assert ctx.validate_data_flow(raw_data, processed_data)
        assert processed_data.shape[0] == 50
        assert processed_data.shape[1] == 4
        assert ctx.adapter.is_compatible_with(ctx.model)
        assert ctx.model.can_accept(processed_data)
```

### Example 3: Scenario-based Testing

**Before**: 복잡한 end-to-end 테스트에서 50+ lines setup

**After**:
```python
def test_full_classification_pipeline(self, classification_scenario):
    scenario = classification_scenario  # All setup done
    
    # 10 lines of pure business logic testing
    pipeline_result = scenario.run_full_pipeline()
    
    assert pipeline_result.model_trained
    assert pipeline_result.evaluation_completed  
    assert pipeline_result.mlflow_logged
    assert scenario.verify_model_quality(min_accuracy=0.7)
    assert scenario.verify_mlflow_artifacts()
    assert scenario.verify_evaluation_metrics()
```

---

## 🚀 Migration Strategy: Zero-Risk Incremental Approach

### Phase 1: Foundation Setup (Week 1)

**새로운 구조 추가 (기존 코드 무변경)**:

1. **Context Classes 생성**:
```bash
mkdir -p tests/fixtures/contexts
mkdir -p tests/fixtures/templates
```

2. **conftest.py 확장**:
```python
# conftest.py에 추가 (기존 fixture들과 함께 공존)
@pytest.fixture
def mlflow_test_context(isolated_temp_directory, settings_builder, test_data_generator):
    return MLflowTestContext(isolated_temp_directory, settings_builder, test_data_generator)

@pytest.fixture  
def component_test_context(isolated_temp_directory, settings_builder, test_data_generator):
    return ComponentTestContext(isolated_temp_directory, settings_builder, test_data_generator)

@pytest.fixture
def classification_scenario(isolated_temp_directory, settings_builder, test_data_generator):
    return ClassificationScenario(isolated_temp_directory, settings_builder, test_data_generator)
```

3. **Template YAML 파일들 생성**:
```yaml
# tests/fixtures/templates/configs/classification_base.yaml
environment:
  name: "test_env"
data_source:
  name: "test_storage" 
  adapter_type: "storage"
mlflow:
  tracking_uri: "{{mlflow_uri}}"
  experiment_name: "{{experiment_name}}"
```

4. **가이드/성능 계측 추가**:
- `tests/fixtures/contexts/README.md`에 컨텍스트 최소 규약·금지사항(엔진 재구현 금지) 명시
- 컨텍스트 초기화 시간 `performance_benchmark`로 계측(권장 임계: 120ms)

### Phase 2: Pilot Testing (Week 2-3)

**A/B 테스팅 방식**:

```python
# 기존 테스트 (변경 없음)
def test_mlflow_experiment_creation_and_tracking(self, isolated_temp_directory, settings_builder):
    # 기존 코드 그대로 유지
    
# 새로운 방식 (같은 파일에 추가)  
def test_mlflow_experiment_creation_and_tracking_v2(self, mlflow_test_context):
    # 새로운 Context 방식
    
# 결과 비교 테스트
def test_compare_old_vs_new_approach(self, isolated_temp_directory, settings_builder, mlflow_test_context):
    # 두 방식의 결과가 동일한지 검증
```

**검증 기준**:
- ✅ 새 방식과 기존 방식 결과 100% 일치
- ✅ 새 방식이 더 짧고 명확한 코드
- ✅ 새 방식이 더 많은 검증 로직 포함 가능
- ✅ 성능 회귀 없음(컨텍스트 init/파이프라인 실행 시간 상한 만족)

### Phase 3: Category-wise Migration (Week 4-6)

**단계별 마이그레이션**:

> Phase 3 상태: ✅ 완료 (v2 확대 + 성능/CI 정비 반영)

1. **MLflow Tests** (Week 4):
   - `test_mlflow_integration.py` 내 11개 테스트
   - 한 번에 하나씩 새 방식으로 마이그레이션
   - 각 테스트마다 결과 일치 확인
   - 완료 내역: v2 케이스(실험 생성/모델 로깅/아티팩트/레지스트리/검색/동등성/베이스라인) 추가 및 A/B 동등성 유지, MLflow 3.x API 호환성(`list_run_infos` → `search_runs`) 반영

2. **Component Interaction Tests** (Week 5):
   - `test_component_interactions.py` 내 10개 테스트
   - Context-based component testing 적용
   - 완료 내역: `ComponentTestContext` 확장(evaluator/preprocessor 노출), v2 흐름(어댑터→모델, 모델→평가자, e2e) 추가

3. **Database Integration Tests** (Week 6):
   - `test_database_integration.py` 내 9개 테스트
   - DatabaseTestContext 적용
   - 완료 내역: v2 읽기 케이스 추가(SQL 가드 적용: SELECT * 금지, LIMIT 권장), 기타 DB 시나리오와 공존 확인

### Phase 4: Validation & Cleanup (Week 7-8)

**최종 검증**:
```bash
# 모든 테스트 실행하여 성공률 확인
pytest tests/integration/ -v
# Expected: 62/62 tests passed (100%)
```

**Cleanup 기준**:
- 새 방식과 기존 방식 결과가 **100% 일치** 확인된 경우에만 기존 코드 제거
- 일치하지 않는 경우 두 방식 공존 유지
- 사용하지 않는 setup 코드만 정리

**Rollback Plan**:
- 임계 카테고리에서 실패/회귀 발생 시, 해당 파일의 v2 테스트를 일시 비활성화하고 기존(v1)만 유지
- 컨텍스트/템플릿 변경은 PR 단위로 격리하여 빠른 revert 가능하도록 유지
- 실패 유형을 리그레션 노트로 기록하고, 컨텍스트 최소 규약 위반 여부 우선 점검

---

## 📊 Expected Benefits

### Quantitative Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Average Test Length** | 25-30 lines | 8-12 lines | **60% reduction** |
| **Setup vs Verification Ratio** | 80:20 | 20:80 | **4x more verification** |
| **Code Duplication** | 30+ repeated patterns | 3-5 centralized contexts | **85% reduction** |
| **New Test Creation Time** | 15-20 minutes | 5-8 minutes | **60% faster** |
| **Test Readability Score** | 6/10 | 9/10 | **50% improvement** |

### Qualitative Improvements

1. **테스트 의도 명확성**: Setup noise 제거로 비즈니스 로직에 집중
2. **유지보수성**: 중앙화된 Context로 한 곳에서 관리
3. **확장성**: 새로운 시나리오 쉽게 추가 가능  
4. **일관성**: 표준화된 패턴으로 개발자 간 일관성 향상
5. **디버깅 용이성**: 문제 발생시 Context 단위로 격리하여 디버깅

### Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| **테스트 실패** | A/B testing으로 결과 일치 확인 후 마이그레이션 |
| **호환성 문제** | 기존 fixture들 절대 변경하지 않고 새 fixture 추가 |
| **성능 저하** | Context 초기화 비용 vs Setup 중복 제거 효과 측정 |
| **학습 비용** | 단계적 도입과 문서화로 점진적 학습 |
| **Context 과기능화** | 컨텍스트 최소 규약 준수, 퍼블릭 API만 호출 |
| **상태/캐시 누수** | 테스트당 새 Factory/Settings 생성, 컨텍스트 재사용 금지 |
| **플레이키(이름/시간 종속)** | uuid 기반 명명, 고정 시드 적용 |
| **초기화 비용 증가** | performance_benchmark로 컨텍스트 init 시간 모니터링(임계 120ms) |

---

## 🎯 Success Metrics

### Technical KPIs
- ✅ **62/62 tests passing** (최우선 지표)
- ✅ **Average test length < 15 lines**
- ✅ **Setup code ratio < 30%**
- ✅ **Zero new flaky tests**
- ✅ **Context init time < 0.12s (p75)**
- ✅ **Zero test state leakage across tests**
- ✅ **Artifact equivalence maintained (metrics/params/signature/schema)**

### Developer Experience KPIs  
- ✅ **New test creation time < 10 minutes**
- ✅ **Test readability score > 8/10** (peer review)
- ✅ **Context adoption rate > 80%** (new tests)
- ✅ **Developer satisfaction score > 4/5**

### CI/Execution Strategy
- **스위트 분리**: `unit`/`integration`/`e2e`를 워크플로 잡으로 분리, 컨텍스트 도입 테스트를 별도 매트릭스에 배치
- **게이팅**: A/B 동등성/성능 상한을 PR 게이트로 추가, 위배 시 머지 차단
- **아티팩트 비교**: MLflow run-level 메트릭/파라미터/시그니처/스키마 요약을 비교하여 동등성 검증 리포트 첨부

#### 병렬 실행 가이드 (권장)
- 로컬/CI: `pytest -n auto --dist=loadscope --durations=15`
- MLflow 파일 스토어는 테스트별 `file://{temp_dir}/mlruns` 고정, 디렉토리 선생성으로 워커 충돌 방지
- 긴 실행 케이스는 병렬화로 분산, 실패 없음을 우선 보장(경고/리포팅은 유지)

---

## 🔄 Implementation Checklist

### Foundation Phase
- [x] Create `tests/fixtures/contexts/` directory
- [x] Implement `MLflowTestContext` class  
- [x] Implement `ComponentTestContext` class
- [x] Implement `DatabaseTestContext` class
- [x] Create YAML templates in `tests/fixtures/templates/`
- [x] Add new fixtures to `conftest.py`
- [x] Verify all existing tests still pass (62/62)
- [x] Standardize MLflow tracking URI and experiment naming
- [x] Enforce deterministic seed usage in contexts
- [x] Add `tests/fixtures/contexts/README.md` (minimal contract & anti-patterns)
- [x] Add performance measurement for context init

### Pilot Phase
- [x] Migrate 2-3 MLflow tests to new approach
- [x] A/B test old vs new approach results
- [x] Measure code length reduction
- [x] Collect developer feedback
- [x] Refine Context implementations based on feedback
- [x] Include performance upper-bound checks in A/B (init + run)
- [x] Add artifact equivalence gate (metrics/params/signature/schema) in CI

### Migration Phase  
- [x] Migrate remaining MLflow tests (8-9 tests)
- [x] Migrate component interaction tests (10 tests)
- [x] Migrate database integration tests (9 tests)
- [x] Migrate remaining integration tests
- [x] Verify 62/62 success rate after each category (병렬 실행 그린 유지)

### Cleanup Phase
- [x] Remove deprecated setup code (MLflow sqlite URIs, empty expected dirs)
- [x] Update documentation (policy and checklist alignment)
- [ ] Create developer guidelines for new Context usage
- [x] Final validation: 62/62 tests passing
- [x] Replace time-based names with uuid-based naming (standardization)

---

## 🎨 Long-term Vision

### 6-Month Goals
- **100% Context Adoption**: 모든 새 테스트가 Context 패턴 사용
- **Template Ecosystem**: 다양한 시나리오용 템플릿 라이브러리 구축  
- **Auto-generation**: 테스트 케이스 자동 생성 도구 개발

### 1-Year Goals  
- **Cross-Project Reusability**: Context 패턴을 다른 프로젝트에도 적용
- **Performance Optimization**: Context 초기화 성능 최적화
- **Advanced Scenarios**: 복잡한 multi-component 시나리오 지원

---

## 💡 Conclusion

### Key Takeaways

1. **현재 구조의 진짜 문제**: Setup overhead가 테스트 코드의 80% 차지
2. **Context-based 해결책**: 책임 분리와 setup 중앙화로 근본 해결  
3. **Zero-risk Migration**: 기존 테스트와 공존하며 점진적 개선
4. **실질적 개선**: 60% 코드 감소, 4배 더 많은 검증 로직

### Final Recommendation

> **DO**: Context-based Architecture로 점진적 마이그레이션  
> **HOW**: A/B testing으로 안전성 확보하며 단계적 적용  
> **GOAL**: 62/62 성공률 유지하면서 더 아름다운 테스트 구조 달성

이 접근법으로 **"현재처럼 다양한 테스트를 완전히 만족하면서도 더 아름답고 책임분리된 tests/ 구조"**를 달성할 수 있습니다.

---

*Generated: 2025-01-XX*  
*Status: Practical Implementation Ready*  
*Next Step: Foundation Phase Implementation*