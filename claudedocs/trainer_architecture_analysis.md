# Trainer Module Architecture Analysis & Complete Refactoring Plan

## 📋 Executive Summary

현재 Trainer 모듈이 과도한 책임을 담당하며, Pipeline과 Pre-factory 역할이 혼재되어 있습니다.
또한 Fetcher가 생성만 되고 실제로 사용되지 않으며, Evaluator도 잘못된 위치에서 실행되고 있습니다.
이 문서는 완전한 아키텍처 분석과 체계적인 리팩토링 계획을 제시합니다.

## 🏗️ 의도된 아키텍처 패턴

```
Pipeline Layer (전체 오케스트레이션)
    ↓
Factory Layer (컴포넌트 생성)
    ↓
Component Layer (개별 책임)
    ├── Registry Pattern: 모듈 자기 등록
    ├── Pre-factory Layer: 내부 모듈 조합 (preprocessor, trainer)
    └── Module: 단일 책임 수행
```

## 🔍 현재 상태 분석

### 1. **Trainer 책임 경계 위반** 🔴

#### 현재 코드 (trainer.py)
```python
def train(self, df, model, fetcher, datahandler, preprocessor, evaluator, context_params):
    # ❌ Pipeline 책임: 데이터 분할
    train_df, test_df = datahandler.split_data(df)
    
    # ❌ Pipeline 책임: 데이터 준비
    X_train, y_train, _ = datahandler.prepare_data(train_df)
    X_test, y_test, _ = datahandler.prepare_data(test_df)
    
    # ❌ Pipeline 책임: 전처리 오케스트레이션
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # ❌ Pipeline 책임: 전처리 데이터 저장 (54-87줄)
    if output_cfg.preprocessed.enabled:
        # 50줄의 저장 로직...
        
    # ✅ Trainer 책임: 학습 & HPO
    if hyperparams.tuning_enabled:
        optimizer = OptunaOptimizer(...)
    else:
        model.fit(X_train, y_train)
    
    # ❌ Pipeline 책임: 평가
    metrics = evaluator.evaluate(trained_model, X_test, y_test)
    
    return trained_model, preprocessor, metrics, training_results
```

**문제점:**
- Trainer가 동일 계층의 컴포넌트들을 오케스트레이션
- 데이터 흐름 제어를 Pipeline 대신 Trainer가 수행
- Pre-factory layer의 범위를 벗어난 책임

### 2. **Fetcher 완전 누락** 🔴

#### 데이터 흐름 분석
```python
# train_pipeline.py
fetcher = factory.create_fetcher()  # ✅ 생성
trainer.train(..., fetcher=fetcher, ...)  # ✅ 전달

# trainer.py
def train(..., fetcher: BaseFetcher, ...):  # ✅ 받음
    # ❌ fetcher 사용 코드 없음!
    # fetcher.fetch() 호출이 어디에도 없음
```

**결과:**
- LOCAL 환경: PassThrough라 문제 없음
- **DEV/PROD 환경: Feature Store 연동 작동 안 함** 🚨
- 피처 증강 단계가 완전히 누락됨

### 3. **Evaluator 위치 문제** 🟡

```python
# trainer.py:107
metrics = evaluator.evaluate(trained_model, X_test, y_test)
```

**문제점:**
- 평가가 Trainer 내부에서 실행
- Pipeline이 평가를 제어할 수 없음
- 평가 전략 변경 시 Trainer 수정 필요

### 4. **중복 기능 & 구조 문제** 🟡

```
src/components/trainer/modules/data_handler.py  # 내부 data handler
src/components/datahandler/                      # 메인 data handler
src/factory/artifact.py                          # 잘못된 위치의 PyfuncWrapper
```

**문제점:**
- 동일 기능이 두 곳에 구현
- PyfuncWrapper는 MLflow 유틸리티인데 factory 아래 위치
- 일관성 없는 모듈 구조

### 5. **직렬화 문제** 🟡

```python
# artifact.py (PyfuncWrapper)
self.trained_fetcher = None  # ❌ 직렬화 문제로 None 처리
```

**원인:**
- FeatureStoreFetcher가 Factory, DB 연결 등 직렬화 불가능한 객체 포함
- MLflow pickle 직렬화 실패

## ✅ 제안하는 리팩토링 방안

### 1. **올바른 데이터 흐름 구현**

```python
# train_pipeline.py
def run_train_pipeline(settings):
    # 1. 컴포넌트 생성
    factory = Factory(settings)
    adapter = factory.create_data_adapter()
    fetcher = factory.create_fetcher()
    datahandler = factory.create_datahandler()
    preprocessor = factory.create_preprocessor()
    model = factory.create_model()
    trainer = factory.create_trainer()
    evaluator = factory.create_evaluator()
    
    # 2. 데이터 로드
    df = adapter.read(source_uri)
    
    # 3. ✨ 피처 증강 (누락된 부분 추가!)
    augmented_df = fetcher.fetch(df, run_mode="train")
    
    # 4. 데이터 분할 & 준비 (Pipeline이 제어)
    train_df, test_df = datahandler.split_data(augmented_df)
    X_train, y_train, additional_train = datahandler.prepare_data(train_df)
    X_test, y_test, additional_test = datahandler.prepare_data(test_df)
    
    # 5. 전처리 (Pipeline이 제어)
    if preprocessor:
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_train_processed, X_test_processed = X_train, X_test
    
    # 6. 전처리 데이터 저장 (Pipeline 책임)
    if settings.config.output.preprocessed.enabled:
        save_preprocessed_data(X_train_processed, X_test_processed, settings)
    
    # 7. 학습 (준비된 데이터만 전달)
    trained_model = trainer.train(
        X_train_processed, 
        y_train,
        X_test_processed,
        y_test,
        model,
        additional_data=additional_train
    )
    
    # 8. 평가 (Pipeline이 제어)
    metrics = evaluator.evaluate(
        trained_model, 
        X_test_processed, 
        y_test, 
        additional_test
    )
    
    # 9. MLflow 저장
    from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper
    pyfunc_wrapper = PyfuncWrapper(
        settings=settings,
        trained_model=trained_model,
        trained_preprocessor=preprocessor,
        fetcher_config=fetcher.get_config(),  # 설정만 저장
        training_results={'metrics': metrics}
    )
    
    return trained_model, metrics
```

### 2. **Trainer 단순화**

```python
# src/components/trainer/trainer.py
class Trainer(BaseTrainer):
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: BaseModel,
        additional_data: Optional[Dict] = None
    ) -> BaseModel:
        """순수한 학습 로직만 담당"""
        
        # 하이퍼파라미터 최적화
        if self.settings.recipe.model.hyperparameters.tuning_enabled:
            optimizer = OptunaOptimizer(self.settings)
            best_model = optimizer.optimize(
                X_train, y_train, X_test, y_test, model
            )
            return best_model
        else:
            # 직접 학습
            task_choice = self.settings.recipe.task_choice
            if task_choice == "causal" and additional_data:
                model.fit(X_train, additional_data['treatment'], y_train)
            elif task_choice == "clustering":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
            
            return model
```

### 3. **PyfuncWrapper 재배치 & 직렬화 해결**

```python
# src/utils/integrations/pyfunc_wrapper.py (이동!)
class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, settings, trained_model, trained_preprocessor, 
                 fetcher_config, training_results):
        # 직렬화 가능한 설정만 저장
        self.settings_dict = self._extract_serializable_settings(settings)
        self.trained_model = trained_model
        self.trained_preprocessor = trained_preprocessor
        self.fetcher_config = fetcher_config  # 설정만 저장 (객체 X)
        self.training_results = training_results
    
    def predict(self, context, model_input, params=None):
        run_mode = params.get("run_mode", "batch")
        
        # 추론 시 fetcher 재생성 (필요한 경우)
        if self.fetcher_config.get('type') == 'feature_store':
            from src.factory import Factory
            factory = Factory(self.settings_dict)
            fetcher = factory.create_fetcher()
            augmented_input = fetcher.fetch(model_input, run_mode)
        else:
            augmented_input = model_input
        
        # 전처리
        if self.trained_preprocessor:
            processed_input = self.trained_preprocessor.transform(augmented_input)
        else:
            processed_input = augmented_input
        
        # 예측
        return self.trained_model.predict(processed_input)
```

### 4. **파일 구조 정리**

```
변경 전:
src/
├── components/
│   └── trainer/
│       ├── trainer.py
│       └── modules/
│           ├── data_handler.py  # 삭제
│           └── optimizer.py
├── factory/
│   └── artifact.py  # 이동

변경 후:
src/
├── components/
│   └── trainer/
│       ├── trainer.py
│       └── modules/
│           └── optimizer.py  # 내부 모듈만 유지
├── utils/
│   └── integrations/
│       ├── mlflow_integration.py
│       └── pyfunc_wrapper.py  # 이동 & 이름 변경
```

## 📋 구현 계획

### Phase 1: 파일 구조 정리
- [ ] `artifact.py` → `utils/integrations/pyfunc_wrapper.py` 이동
- [ ] `trainer/modules/data_handler.py` 삭제
- [ ] Import 경로 업데이트

### Phase 2: Pipeline 강화
- [ ] `fetcher.fetch()` 호출 추가
- [ ] 데이터 분할/준비 로직을 pipeline으로 이동
- [ ] 전처리 오케스트레이션을 pipeline으로 이동
- [ ] 전처리 데이터 저장 로직을 pipeline으로 이동
- [ ] `evaluator.evaluate()`를 pipeline으로 이동

### Phase 3: Trainer 단순화
- [ ] `train()` 메서드 시그니처 변경
- [ ] Fetcher, DataHandler, Preprocessor, Evaluator 의존성 제거
- [ ] 순수 학습 로직만 유지
- [ ] OptunaOptimizer와의 통합 유지

### Phase 4: 직렬화 문제 해결
- [ ] PyfuncWrapper에서 fetcher_config만 저장
- [ ] `predict()` 메서드에서 fetcher 재생성 로직 구현
- [ ] Inference pipeline에서 fetcher 재생성 지원

### Phase 5: 테스트 & 검증
- [ ] 단위 테스트 업데이트
- [ ] 통합 테스트 실행
- [ ] Feature Store 연동 테스트 (DEV/PROD)
- [ ] MLflow 모델 저장/로드 테스트

## 🎯 기대 효과

### 1. **명확한 책임 분리**
| 컴포넌트 | 책임 | 의존성 |
|---------|-----|--------|
| **Pipeline** | 전체 워크플로우 오케스트레이션 | 모든 컴포넌트 |
| **Trainer** | 모델 학습 & HPO만 | OptunaOptimizer (내부) |
| **DataHandler** | 데이터 분할 & 준비 | 없음 |
| **Fetcher** | 피처 증강 | FeatureStore (옵션) |
| **Preprocessor** | 전처리 스텝 조합 | PreprocessorSteps (내부) |
| **Evaluator** | 모델 평가 | 없음 |

### 2. **완전한 데이터 흐름**
```
Load → Fetch → Split → Prepare → Preprocess → Train → Evaluate → Save
  ↑      ↑       ↑        ↑          ↑          ↑        ↑        ↑
Pipeline이 모든 단계를 명시적으로 제어
```

### 3. **Feature Store 연동 복구**
- DEV/PROD 환경에서 Feature Store 피처 증강 정상 작동
- 추론 시에도 동일한 피처 증강 보장

### 4. **유지보수성 향상**
- 코드 중복 제거
- 일관된 모듈 구조
- 명확한 파일 위치
- 테스트 용이성 증가

### 5. **확장성 개선**
- 새로운 Trainer 구현 시 인터페이스가 단순
- 평가 전략 변경이 Pipeline 레벨에서 가능
- 피처 증강 전략 교체 용이

## 💡 추가 권장사항

### 1. **인터페이스 정의 강화**
```python
# src/interface/base_trainer.py
class BaseTrainer(ABC):
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: BaseModel,
        **kwargs
    ) -> BaseModel:
        """순수한 학습 인터페이스"""
        pass
```

### 2. **설정 검증 강화**
- Pipeline 시작 시 모든 컴포넌트 설정 검증
- Fetcher 타입과 환경 일치 검증
- 데이터 스키마 일관성 검증

### 3. **로깅 표준화**
- 각 컴포넌트 경계에서 명확한 로깅
- 데이터 shape 변화 추적
- 성능 메트릭 기록

## 📊 리스크 & 완화 방안

| 리스크 | 영향도 | 완화 방안 |
|--------|-------|----------|
| 기존 테스트 깨짐 | 높음 | 단계적 리팩토링, 테스트 먼저 수정 |
| Feature Store 연동 실패 | 중간 | DEV 환경에서 충분한 테스트 |
| 직렬화 문제 재발 | 낮음 | 설정만 저장하는 패턴 검증 |
| 성능 저하 | 낮음 | 벤치마크 테스트 수행 |

## 🚀 결론

이 리팩토링을 통해:
1. **의도한 아키텍처 패턴을 완전히 구현**
2. **누락된 Fetcher 기능 복구**
3. **명확한 책임 분리로 유지보수성 향상**
4. **Feature Store 연동 정상화로 DEV/PROD 환경 지원**

모든 컴포넌트가 단일 책임 원칙을 준수하고, Pipeline이 명확하게 전체 흐름을 제어하는 깔끔한 아키텍처가 완성됩니다.