# 🧪 Phase 3 테스트 개발 계획서 - Modern ML Pipeline

## 📊 현재 상황 분석

### Ultra Think 분석 결과 요약
**분석 일자**: 2025-09-06  
**분석 방법**: Sequential MCP를 통한 심층 구조 분석  
**현재 커버리지**: **29%** (목표 60%와 31%p 차이)

### 테스트 현황 Summary
| 항목 | 현재 상태 | 목표 |
|------|-----------|------|
| **총 테스트 파일** | 34개 | - |
| **총 테스트 케이스** | 419개 | - |
| **통과 테스트** | 395개 (94.3%) | 100% |
| **스킵 테스트** | 24개 (Feast 미설치) | 0개 |
| **커버리지** | 29% | 60% |

### Phase별 완료 현황
- ✅ **Phase 1**: 기초 인프라 구축 (완료)
- ✅ **Phase 2**: Core 단위 테스트 (완료)  
- 🔄 **Phase 3**: Component 단위 테스트 (부분 완료)
- ⏳ **Phase 4**: 통합 테스트 (대기 중)
- ⏳ **Phase 5**: E2E 및 CLI 테스트 (대기 중)

## 🎯 Phase 3 미완료 분석

### 완료된 컴포넌트 ✅
- **Adapter**: 4개 테스트 파일, Registry 테스트 포함
- **Fetcher**: 3개 테스트 파일, Registry 테스트 포함  
- **Evaluator**: 5개 테스트 파일, Registry 테스트 포함

### 미완료 컴포넌트 ❌

#### 1. Trainer 컴포넌트 (최고 우선순위)
**누락된 테스트 파일:**
```
tests/unit/components/test_trainer/
├── test_trainer_registry.py     ✅ 완료
├── test_trainer.py              ❌ 누락 (핵심)
├── test_data_handler.py         ❌ 누락
└── test_optimizer.py            ❌ 누락
```

**테스트 대상 소스 코드:**
```
src/components/trainer/
├── modules/trainer.py           ← 메인 Trainer 클래스
├── modules/data_handler.py      ← 데이터 분할/준비
├── modules/optimizer.py         ← Optuna 통합
└── registry.py                  ✅ 테스트 완료
```

#### 2. Preprocessor 컴포넌트 (두 번째 우선순위)
**누락된 테스트 파일:**
```
tests/unit/components/test_preprocessor/
├── test_preprocessor_step_registry.py  ✅ 완료
├── test_preprocessor.py                ❌ 누락 (핵심)
├── test_scaler.py                      ❌ 누락
├── test_encoder.py                     ❌ 누락
├── test_imputer.py                     ❌ 누락
├── test_feature_generator.py           ❌ 누락
├── test_discretizer.py                 ❌ 누락
└── test_missing.py                     ❌ 누락
```

**테스트 대상 소스 코드:**
```
src/components/preprocessor/
├── preprocessor.py              ← 메인 Preprocessor 클래스
├── modules/scaler.py           ← StandardScaler, MinMaxScaler 등
├── modules/encoder.py          ← OneHotEncoder, OrdinalEncoder 등
├── modules/imputer.py          ← SimpleImputer
├── modules/feature_generator.py ← PolynomialFeatures 등
├── modules/discretizer.py      ← KBinsDiscretizer
├── modules/missing.py          ← MissingIndicator
└── registry.py                 ✅ 테스트 완료
```

## 🚀 Phase 3 개발 로드맵

### Week 1: Trainer 컴포넌트 (5일)

#### Day 1: test_data_handler.py
**개발 목표**: 데이터 분할 및 준비 로직 테스트
**예상 시간**: 8시간
**핵심 테스트 케이스**:
```python
class TestDataHandler:
    def test_split_data_stratified_classification(self):
        # 분류 task에서 stratified split 검증
        
    def test_split_data_causal_treatment_stratify(self):
        # Causal task에서 treatment 기반 stratify
        
    def test_prepare_training_data_feature_auto_selection(self):
        # feature_columns=None일 때 자동 선택 로직
        
    def test_prepare_training_data_task_specific_processing(self):
        # Task별 (classification/regression/clustering/causal) 데이터 처리
        
    def test_edge_cases_small_dataset_no_stratify(self):
        # 소규모 데이터셋에서 stratify 불가능한 경우
```

#### Day 2: test_optimizer.py  
**개발 목표**: Optuna 하이퍼파라미터 최적화 테스트
**예상 시간**: 8시간
**핵심 테스트 케이스**:
```python
class TestOptunaOptimizer:
    def test_optimizer_initialization_with_tuning_config(self):
        # Recipe tuning 설정에 따른 초기화
        
    def test_optimize_study_creation_and_execution(self, mock_optuna):
        # Study 생성 및 최적화 실행 Mock
        
    def test_hyperparameter_space_definition(self):
        # tunable 파라미터 공간 정의 검증
        
    def test_objective_function_execution(self):
        # 목적 함수 실행 및 점수 반환
        
    def test_optimization_error_handling(self):
        # 최적화 실패, 타임아웃 등 에러 처리
```

#### Day 3-4: test_trainer.py (2일)
**개발 목표**: 메인 Trainer 클래스 완전 테스트  
**예상 시간**: 16시간 (가장 복잡)
**핵심 테스트 케이스**:
```python
class TestTrainer:
    def test_trainer_initialization_with_factory_provider(self):
        # Factory Provider 패턴 테스트
        
    def test_train_with_optuna_enabled(self, mock_optuna, mock_factory):
        # Optuna 활성화 시 전체 훈련 플로우
        
    def test_train_with_fixed_hyperparameters(self, mock_factory):
        # 고정 하이퍼파라미터 훈련 플로우
        
    def test_train_task_specific_workflows(self):
        # Task별 (classification/regression/clustering/causal) 워크플로우
        
    def test_single_training_iteration_data_leakage_prevention(self):
        # Optuna 튜닝 시 Data Leakage 방지를 위한 3단계 분할
        
    def test_fit_model_task_specific_patterns(self):
        # Task별 모델 fitting 패턴 (causal의 treatment 파라미터 등)
        
    def test_training_methodology_metadata_generation(self):
        # 훈련 방법론 메타데이터 생성 검증
        
    def test_component_orchestration_error_handling(self):
        # 컴포넌트 간 오케스트레이션 에러 처리
```

#### Day 5: 통합 테스트 및 디버깅
**개발 목표**: Trainer 관련 모든 테스트 안정화
**예상 시간**: 8시간
**주요 작업**:
- 3개 테스트 파일 간 상호작용 검증
- Mock 패턴 일관성 확보
- 성능 최적화 (각 테스트 1초 이내)
- Registry 격리 안정성 검증

### Week 2: Preprocessor 컴포넌트 (2일)

#### Day 1: test_preprocessor.py + 주요 모듈 3개
**개발 목표**: 메인 Preprocessor + 핵심 모듈 테스트
**예상 시간**: 8시간
**주요 작업**:
```python
# test_preprocessor.py - 메인 클래스
class TestPreprocessor:
    def test_preprocessor_initialization_with_settings(self):
    def test_pipeline_creation_from_recipe_steps(self):
    def test_fit_transform_pipeline_execution(self):
    def test_dynamic_step_configuration(self):

# test_scaler.py - 가장 중요한 전처리
class TestScalerSteps:
    def test_standard_scaler_step(self):
    def test_min_max_scaler_step(self):
    def test_robust_scaler_step(self):

# test_encoder.py - 두 번째 중요
class TestEncoderSteps:
    def test_one_hot_encoder_step(self):
    def test_ordinal_encoder_step(self):
    def test_catboost_encoder_step(self):

# test_imputer.py - 세 번째 중요
class TestImputerSteps:
    def test_simple_imputer_with_strategies(self):
    def test_imputer_error_handling(self):
```

#### Day 2: 나머지 모듈 3개 + 통합 테스트
**개발 목표**: 남은 모듈들 + 전체 통합
**예상 시간**: 8시간
**주요 작업**:
```python
# test_feature_generator.py
class TestFeatureGeneratorSteps:
    def test_polynomial_features_step(self):
    def test_tree_based_feature_generator_step(self):

# test_discretizer.py  
class TestDiscretizerSteps:
    def test_kbins_discretizer_step(self):

# test_missing.py
class TestMissingSteps:
    def test_missing_indicator_step(self):

# 통합 테스트
class TestPreprocessorIntegration:
    def test_full_preprocessing_pipeline(self):
    def test_preprocessor_with_all_step_types(self):
```

## 📈 예상 효과

### 커버리지 향상 예측
| 구성 요소 | 예상 커버리지 기여 | 누적 목표 |
|-----------|-------------------|-----------|
| **현재 베이스라인** | - | 29% |
| **+ Trainer 완료** | +15%p | 44% |
| **+ Preprocessor 완료** | +12%p | **56%** |
| **목표 달성률** | - | **93%** (목표 60%) |

### 테스트 품질 개선
- **총 테스트 케이스**: 419개 → **500+개** (80+ 추가)
- **실행 속도**: 각 테스트 1초 이내 유지
- **Registry 격리**: 완전한 테스트 간 독립성 보장
- **Mock 패턴**: 일관된 Factory Provider Mock 패턴

## 🛠️ 구현 전략

### 1. Mock 전략

#### Factory Provider Mock 패턴
```python
@pytest.fixture
def mock_factory_provider(mock_factory):
    """Trainer에서 사용할 Factory Provider Mock"""
    def factory_provider():
        return mock_factory
    return factory_provider

@pytest.fixture  
def trainer_with_mocked_factory(test_settings, mock_factory_provider):
    """Factory Provider가 주입된 Trainer"""
    return Trainer(settings=test_settings, factory_provider=mock_factory_provider)
```

#### Optuna Study Mock 패턴
```python
@pytest.fixture
def mock_optuna_study():
    """Optuna Study 객체 Mock"""
    with patch('optuna.create_study') as mock_create_study:
        mock_study = MagicMock()
        mock_study.optimize.return_value = None
        mock_study.best_params = {'n_estimators': 100, 'max_depth': 10}
        mock_study.best_value = 0.95
        mock_create_study.return_value = mock_study
        yield mock_study
```

### 2. 데이터 Builder 활용

#### Trainer 테스트용 데이터
```python
# DataFrameBuilder 확장
class TrainerDataBuilder:
    @staticmethod
    def build_train_test_split_data(task_type="classification", n_samples=100):
        """Train/Test 분할 테스트용 데이터"""
        if task_type == "classification":
            return DataFrameBuilder.build_classification_data(n_samples)
        elif task_type == "causal":
            return DataFrameBuilder.build_causal_data(n_samples)
        # ... 기타 task_type
    
    @staticmethod
    def build_optuna_test_scenario():
        """Optuna 최적화 테스트 시나리오"""
        return {
            'train_df': DataFrameBuilder.build_classification_data(80),
            'val_df': DataFrameBuilder.build_classification_data(20),
            'hyperparameter_space': {
                'n_estimators': {'type': 'int', 'range': [10, 100]},
                'max_depth': {'type': 'int', 'range': [3, 20]}
            }
        }
```

### 3. 성능 최적화

#### 테스트 실행 속도 관리
```python
@pytest.mark.timeout(1)  # 각 테스트 1초 이내
class TestTrainerPerformance:
    """성능 중심 테스트 케이스들"""
    
    def test_trainer_initialization_speed(self):
        """Trainer 초기화 속도 검증"""
        start_time = time.time()
        trainer = Trainer(settings=test_settings, factory_provider=mock_factory_provider)
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # 100ms 이내
```

## ⚠️ 리스크 관리

### 높은 리스크 (High) - 즉시 대응 필요

#### 1. Factory Provider Mock 복잡도
**문제**: Trainer의 `factory_provider()` 콜백 패턴이 기존 Mock 방식과 상이  
**대응**: 
- 전용 `mock_factory_provider` fixture 개발
- 기존 `mock_factory`와의 호환성 보장
- 의존성 주입 패턴 테스트 강화

#### 2. Optuna Study 객체 Mock
**문제**: Optuna의 복잡한 Study/Trial 구조  
**대응**:
- Study 생성부터 최적화 실행까지 전 과정 Mock
- 실제 최적화 로직 없이 결과값만 Mock
- 타임아웃 및 예외 상황 Mock

### 중간 리스크 (Medium) - 주의 깊게 모니터링

#### 3. 데이터 분할 로직 검증
**문제**: Train/Val/Test 3단계 분할의 통계적 검증  
**대응**:
- 분할 비율 정확성 검증 (80%/16%/20%)
- Stratify 로직 검증 (클래스 분포 균등성)
- Edge case 처리 (소수 클래스, 작은 데이터셋)

#### 4. Task별 분기 로직 테스트
**문제**: 4개 Task Type의 각각 다른 처리 로직  
**대응**:
- Task별 전용 테스트 케이스 개발
- Parameter 검증 강화 (causal의 treatment 등)
- 크로스 Task 호환성 검증

### 낮은 리스크 (Low) - 일반적 주의

#### 5. Preprocessor Pipeline 순서
**문제**: 다중 전처리 단계의 순서 의존성  
**대응**:
- Pipeline 실행 순서 검증
- 각 단계별 결과 검증
- 전체 Pipeline 통합 테스트

## 🔍 품질 보증

### 코드 품질 체크리스트
- [ ] **타입 힌트**: 모든 함수/메서드에 완전한 타입 힌트
- [ ] **Docstring**: Google Style Docstring 필수
- [ ] **AAA 패턴**: Arrange-Act-Assert 구조 준수
- [ ] **테스트 이름**: 의도가 명확한 서술형 네이밍
- [ ] **Mock 격리**: 각 테스트의 완전한 독립성
- [ ] **에러 처리**: 모든 예외 상황 커버
- [ ] **Edge Case**: 경계값 및 특수 상황 테스트

### 성능 기준
- [ ] **실행 속도**: 각 테스트 1초 이내
- [ ] **메모리 사용**: Registry 초기화 후 정리
- [ ] **병렬 실행**: pytest-xdist 호환성
- [ ] **재현성**: 동일한 결과 보장

## 📊 성공 기준

### Phase 3 완료 조건
- ✅ **커버리지 55% 이상** (목표 60%의 92%)
- ✅ **모든 테스트 통과** (Fail 0개)
- ✅ **성능 기준 충족** (평균 실행시간 1초 이내)
- ✅ **CI/CD 그린** (GitHub Actions 통과)

### 정량적 목표
| 메트릭 | 현재 | 목표 | 달성률 |
|--------|------|------|--------|
| **커버리지** | 29% | 55%+ | 190%+ |
| **테스트 케이스** | 419개 | 500+개 | 120%+ |
| **통과율** | 94.3% | 100% | 106%+ |
| **평균 실행시간** | 2.01초 | <3초 | 133% |

## 🎯 Next Steps

### 즉시 시작 가능한 작업 (Day 1)
1. **test_data_handler.py** 파일 생성
2. **TrainerDataBuilder** 클래스 구현  
3. **mock_factory_provider** fixture 개발
4. 첫 번째 테스트 케이스 구현

### 개발 시작 명령어
```bash
# 개발 환경 설정
cd /Users/wooshikwon/Desktop/github_wooshikwon/modern-ml-pipeline
uv sync --all-extras

# Phase 3 테스트 구조 생성
mkdir -p tests/unit/components/test_trainer
touch tests/unit/components/test_trainer/test_trainer.py
touch tests/unit/components/test_trainer/test_data_handler.py  
touch tests/unit/components/test_trainer/test_optimizer.py

# 첫 번째 테스트 실행
uv run pytest tests/unit/components/test_trainer/test_data_handler.py -v

# 커버리지 실시간 모니터링
uv run pytest --cov=src.components.trainer --cov-report=term-missing tests/unit/components/test_trainer/
```

## 📚 참고 문서

### 테스트 패턴 참고
- **기존 성공 사례**: `tests/unit/components/test_evaluator/test_classification_evaluator.py`
- **Factory Mock 패턴**: `tests/conftest.py` - `mock_factory` fixture
- **Builder 패턴**: `tests/helpers/builders.py`
- **Registry 격리**: `tests/conftest.py` - `clean_registries` fixture

### 소스 코드 분석 대상
- **Trainer 메인**: `src/components/trainer/modules/trainer.py:15-151`
- **Data Handler**: `src/components/trainer/modules/data_handler.py:10-142`
- **Optuna 통합**: `src/components/trainer/modules/optimizer.py`
- **Preprocessor 메인**: `src/components/preprocessor/preprocessor.py`

---

**📅 계획서 작성일**: 2025-09-06  
**🎯 Phase 3 목표 완료일**: 2025-09-13 (7일)  
**📈 예상 커버리지 달성**: 56% (목표 60%의 93%)  
**⚡ Ultra Think 분석 기반**: Sequential MCP 심층 구조 분석 완료