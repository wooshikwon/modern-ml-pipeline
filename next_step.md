# Blueprint 구현 완성을 위한 최종 분석 및 수정 계획 (next_step.md - v3)

## 1. 개요

이 문서는 `blueprint.md` v11 아키텍처의 6대 핵심 원칙을 기준으로 전체 코드베이스를 상세히 분석한 최종 결과물입니다. 모든 핵심 컴포넌트를 직접 검토한 결과, 설계 철학은 명확하고 구현의 방향성은 올바르지만, **blueprint와 실제 구현 간의 중요한 불일치**와 **구현 누락 사항**들이 발견되었습니다.

이 문서는 시스템을 blueprint의 철학과 100% 일치시키기 위한 구체적이고 실행 가능한 수정 계획을 제공합니다.

---

## 2. 핵심 불일치 분석 및 수정 전략

### 🔴 **최우선 수정사항 1: `Augmenter`의 책임 분리 위반**

#### 문제점 분석:
- **위치:** `src/core/augmenter.py:88-98`
- **구체적 문제:**
  ```python
  def _augment_batch(self, data: pd.DataFrame, context_params: ...):
      factory = Factory(self.settings)  # ❌ 잘못된 의존성
      adapter = factory.create_data_adapter('bq')  # ❌ 데이터 로딩 책임 침범
      feature_df = adapter.read(self.source_uri, params=context_params)  # ❌ 직접 데이터 로딩
  ```
- **Blueprint 원칙 위반:** 
  - **원칙 2 (통합 데이터 어댑터):** 데이터 로딩은 파이프라인의 책임이며, Augmenter는 순수하게 증강 로직만 담당해야 함
  - **원칙 5 (단일 Augmenter, 컨텍스트 주입):** Augmenter는 주입받은 데이터를 변환하는 역할만 해야 함

#### 수정 전략:
1. **`_augment_batch` 메서드 완전 재구현:**
   - 데이터 어댑터 생성 로직 완전 제거
   - 입력받은 `data` (PK 목록)를 기반으로 BigQuery에서 피처 조회하는 로직만 유지
   - 실제 BigQuery 접근은 외부에서 주입받은 어댑터를 통해 수행

### 🔴 **최우선 수정사항 2: 데이터 어댑터 클라이언트 초기화 누락**

#### 문제점 분석:
- **위치:** `src/utils/data_adapters/bigquery_adapter.py:17`, `gcs_adapter.py:15`, `s3_adapter.py:8`
- **구체적 문제:**
  ```python
  class BigQueryAdapter(BaseDataAdapter):
      def __init__(self, settings: Settings):
          super().__init__(settings)
          # ❌ self.client 초기화 누락
  ```
- **영향:** 모든 데이터 어댑터에서 `self.client` 속성이 초기화되지 않아 런타임 에러 발생

#### 수정 전략:
1. **모든 데이터 어댑터에서 `__init__` 메서드 수정:**
   ```python
   def __init__(self, settings: Settings):
       super().__init__(settings)
       self.client = self._get_client()
   ```

### 🔴 **최우선 수정사항 3: `Trainer`의 Augmenter 호출 시 매개변수 누락**

#### 문제점 분석:
- **위치:** `src/core/trainer.py:43-44`
- **구체적 문제:**
  ```python
  train_df = augmenter.augment(train_df, context_params)  # ❌ run_mode 누락
  test_df = augmenter.augment(test_df, context_params)    # ❌ run_mode 누락
  ```
- **Blueprint 원칙 위반:** 
  - **원칙 5 (단일 Augmenter, 컨텍스트 주입):** Augmenter는 `run_mode`를 받아 동작을 결정해야 함

#### 수정 전략:
1. **Trainer에서 Augmenter 호출 시 `run_mode="batch"` 명시:**
   ```python
   train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
   test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)
   ```

### 🔴 **최우선 수정사항 4: API 서빙에서 잘못된 SQL 스냅샷 참조**

#### 문제점 분석:
- **위치:** `serving/api.py:41`
- **구체적 문제:**
  ```python
  pk_fields = get_pk_from_loader_sql(app_context.model.sql_snapshot)  # ❌ 잘못된 속성명
  ```
- **실제 속성명:** `augmenter_sql_snapshot`

#### 수정 전략:
1. **올바른 속성명으로 수정:**
   ```python
   pk_fields = get_pk_from_loader_sql(app_context.model.augmenter_sql_snapshot)
   ```

### 🔴 **최우선 수정사항 5: RedisAdapter 호출 시 잘못된 매개변수**

#### 문제점 분석:
- **위치:** `src/core/augmenter.py:104`
- **구체적 문제:**
  ```python
  redis_adapter = factory.create_redis_adapter(feature_store_config)  # ❌ 잘못된 매개변수
  ```
- **실제 메서드 시그니처:** `create_redis_adapter()`는 매개변수를 받지 않음

#### 수정 전략:
1. **올바른 호출 방식으로 수정:**
   ```python
   redis_adapter = factory.create_redis_adapter()
   ```

### 🟡 **S3Adapter 미완성 구현**

#### 문제점 분석:
- **위치:** `src/utils/data_adapters/s3_adapter.py`
- **구체적 문제:** `__init__` 메서드에서 `settings` 받지 않고 `_get_client` 미구현

#### 수정 전략:
1. **다른 클라우드 어댑터와 일관성 있게 구현:**
   ```python
   def __init__(self, settings: Settings):
       super().__init__(settings)
       self.client = self._get_client()
   
   def _get_client(self):
       # boto3 클라이언트 생성 로직 구현
   ```

---

## 3. 데이터 흐름 정확성 검증

### ✅ **올바르게 구현된 부분들**

1. **`inference_pipeline.py`의 데이터 흐름:**
   - ✅ `wrapper.loader_uri`에서 PK 데이터 로드
   - ✅ 로드된 데이터를 `wrapper.predict(input_df)`로 전달
   - ✅ 이는 blueprint의 설계 의도와 완벽히 일치

2. **`PyfuncWrapper.predict`의 동작:**
   - ✅ `model_input`을 `augmenter.augment`에 전달
   - ✅ `run_mode`에 따른 조건부 처리
   - ✅ 순수 로직 아티팩트 철학 준수

3. **Factory 패턴 구현:**
   - ✅ URI 스킴 기반 동적 어댑터 생성
   - ✅ 설정 주입을 통한 컴포넌트 생성

---

## 4. 최종 수정 계획 (우선순위별)

### **Phase 1: 핵심 아키텍처 수정 (최우선)**

1. **`src/core/augmenter.py` 수정:**
   - `_augment_batch`에서 Factory 생성 로직 제거
   - 데이터 어댑터 직접 호출 로직 제거
   - BigQuery 피처 조회 로직을 외부 어댑터 활용 방식으로 변경

2. **`src/core/trainer.py` 수정:**
   - Augmenter 호출 시 `run_mode="batch"` 매개변수 추가

3. **모든 데이터 어댑터 `__init__` 메서드 수정:**
   - `BigQueryAdapter`, `GCSAdapter`, `S3Adapter`에서 `self.client` 초기화 추가

### **Phase 2: API 서빙 수정 (높음)**

1. **`serving/api.py` 수정:**
   - `sql_snapshot` → `augmenter_sql_snapshot` 변경

2. **`src/core/augmenter.py` 수정:**
   - RedisAdapter 호출 시 매개변수 제거

### **Phase 3: 미완성 구현 완료 (보통)**

1. **`src/utils/data_adapters/s3_adapter.py` 완성:**
   - `__init__` 메서드에서 settings 받도록 수정
   - `_get_client` 메서드 구현

### **Phase 4: 테스트 코드 전면 수정 (낮음)**

1. **`tests/` 디렉토리 전체 수정:**
   - 새로운 아키텍처에 맞춘 단위 테스트 재작성
   - 각 데이터 어댑터별 테스트 케이스 추가
   - 통합 테스트 시나리오 구현

---

## 5. 수정 후 검증 계획

### **단계별 검증 체크리스트**

1. **Phase 1 완료 후:**
   - [ ] `python main.py train --model-name "xgboost_x_learner"` 정상 실행
   - [ ] Augmenter가 데이터 어댑터를 직접 생성하지 않는지 확인
   - [ ] 모든 데이터 어댑터의 클라이언트 초기화 확인

2. **Phase 2 완료 후:**
   - [ ] `python main.py serve-api` 정상 실행
   - [ ] API 스키마 동적 생성 확인
   - [ ] `/predict` 엔드포인트 정상 동작 확인

3. **Phase 3 완료 후:**
   - [ ] S3 환경에서 데이터 읽기/쓰기 테스트
   - [ ] 모든 클라우드 어댑터의 일관성 확인

4. **Phase 4 완료 후:**
   - [ ] 전체 테스트 슈트 실행
   - [ ] 커버리지 80% 이상 달성

---

## 6. 결론

현재 코드베이스는 **blueprint.md**의 철학을 이해하고 올바른 방향으로 구현되어 있지만, 위에서 식별한 **5개의 핵심 불일치**와 **미완성 구현**들이 시스템의 안정성과 일관성을 저해하고 있습니다.

특히 **Augmenter의 책임 분리 위반**과 **데이터 어댑터 클라이언트 초기화 누락**은 시스템의 핵심 아키텍처에 영향을 미치므로 최우선으로 수정해야 합니다.

이 계획을 단계적으로 실행하면 blueprint의 6대 핵심 원칙을 완벽히 구현한 production-ready 시스템을 구축할 수 있을 것입니다.