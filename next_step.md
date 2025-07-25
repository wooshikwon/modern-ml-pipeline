# **Modern ML Pipeline: 전체 테스트 및 시스템 안정성 확보 마스터 플랜**

## **Phase 1: 기반 구축 및 계약 확립 (완료)**

*   **성과:** `mmp-local-dev` 환경을 최신 버전으로 안정화하고, `modern-ml-pipeline`과의 "코드로서의 계약"을 확립함. 의존성 주입 원칙을 적용하여 시스템의 핵심 아키텍처를 개선함.
*   **상태:** **완료.**

---

## **Phase 2: 테스트 스위트 전체 현대화 (현재 진행 임무)**

### **1. 임무 목표 (Mission Objective)**

메인 시스템과 테스트 코드 간의 **"아키텍처 드리프트(Architecture Drift)"** 현상을 완전히 해소하고, 프로젝트의 모든 테스트 코드를 **`Blueprint v17.0`의 철학**에 완벽하게 부합하도록 전면 재구축한다. 이를 통해 **테스트 커버리지를 복원 및 확장**하고, **시스템의 안정성과 신뢰도를 100% 확보**하여 `Phase 3`로 나아갈 수 있는 견고한 기반을 마련한다.

### **2. 핵심 실행 원칙 (Core Principles)**

*   **의도는 계승, 구현은 현대화:** 낡은 테스트 코드의 "핵심 검증 의도"는 유지하되, 현재 아키텍처(`Settings` 객체 주입, 동적 로딩)에 맞게 구현 방식을 완전히 변경한다.
*   **Fixture 중심 설계:** `tests/conftest.py`에 정의된 `local_test_settings`, `dev_test_settings`를 모든 테스트의 일관된 설정 주입 통로로 사용한다.
*   **Mock 최소화, 실제 객체 최대화:** 단위 테스트에서는 외부 I/O만 Mocking하고, 파이프라인/API 테스트는 실제 객체를 사용하는 E2E 테스트로 전환하여 검증 신뢰도를 극대화한다.

### **3. 상세 실행 계획 (Actionable Steps)**

*   **Step 1: 테스트 기반 재설계 (`tests/conftest.py`)**
*   **Step 2: 핵심 로직 단위 테스트 현대화 (`tests/core/`)**
*   **Step 3: End-to-End 테스트 전환 (`tests/pipelines`, `tests/serving`)**
*   **Step 4: 나머지 테스트 현대화 및 자산 정리 (`tests/utils`, `tests/settings`, `tests/environments`, `tests/integration`, `tests/models`)**
*   **최종 검증:** `APP_ENV=dev .venv/bin/python -m pytest -v` 실행하여 **모든 테스트 통과(PASS)** 확인.

---

## **Phase 3: Feature Store 심층 테스트 (다음 단계)**

### **1. 임무 목표**

`Phase 2`에서 복원된 테스트 기반 위에서, `Feast`를 활용하는 Feature Store의 모든 기능적 측면을 심층적으로 검증한다. 데이터의 정확성, 시점 일관성, 온라인/오프라인 서빙의 패리티(parity)를 100% 보장한다.

### **2. 상세 실행 계획**

*   **Step 1: 데이터 수집(Ingestion) 및 정확성 검증:**
    *   `mmp-local-dev`의 `seed-features.sql` 데이터를 `Feast`가 올바르게 `materialize` 하는지 검증.
    *   `PostgreSQL`(오프라인)과 `Redis`(온라인)에 저장된 피처 값들이 원본 데이터와 100% 일치하는지 확인.

*   **Step 2: 시계열(Point-in-time) 정확성 검증:**
    *   `get_historical_features` 호출 시, `event_timestamp`를 기준으로 정확한 과거 시점의 피처를 조회하는지 검증. (Data Leakage 방지 핵심)
    *   "Time Travel" 쿼리를 통해, 과거 특정 시점의 피처 값이 올바르게 반환되는지 테스트 케이스 추가.

*   **Step 3: 온라인/오프라인 패리티(Parity) 검증:**
    *   동일한 엔티티 키에 대해 오프라인 저장소에서 조회한 피처와 온라인 저장소에서 조회한 피처가 **데이터 타입과 값 모두에서 완벽하게 일치**하는지 검증하는 테스트를 자동화.

---

## **Phase 4: 성능 및 안정성 테스트 (미래 계획)**

### **1. 임무 목표**

시스템이 대용량 데이터와 높은 부하 상황에서도 안정적으로 동작하며, `Blueprint`에 정의된 성능 목표를 달성하는지 검증한다. 장애 상황에 대한 복구 능력과 자원 관리 효율성을 확인한다.

### **2. 상세 실행 계획**

*   **Step 1: 성능 벤치마크 테스트:**
    *   `pytest-benchmark`를 사용하여 `local` 및 `dev` 환경에서 학습/추론 파이프라인의 실행 시간을 측정하고, `next_step.md`에 정의된 목표(예: LOCAL 3분, DEV 5분 이내)를 충족하는지 검증.
    *   대용량(100만 건 이상) 데이터 처리 시 메모리 사용량과 처리 시간을 측정.

*   **Step 2: 부하 테스트 (API):**
    *   `locust`와 같은 도구를 사용하여 `/predict` 엔드포인트에 동시 요청 부하를 발생시켜, 초당 처리량(TPS), 응답 시간, 에러율을 측정.

*   **Step 3: 재현성(Reproducibility) 테스트:**
    *   동일한 `Settings`와 데이터로 학습을 여러 번 반복 실행했을 때, 생성되는 모델 아티팩트와 평가 메트릭이 100% 동일한지 검증.

---

## **Phase 5: CI/CD 자동화 (최종 목표)**

### **1. 임무 목표**

`GitHub Actions`를 통해 "테스트 스위트 전체 현대화" 과정에서 재구축된 모든 테스트를 자동화하여, 코드 변경 시 시스템의 안정성이 지속적으로 검증되는 완전한 CI/CD 파이프라인을 구축한다.

### **2. 상세 실행 계획**

*   **Step 1: 단위/통합 테스트 자동화:**
    *   `Push` 또는 `Pull Request`가 발생할 때마다, `pytest -m "not e2e"`를 실행하여 빠른 단위/통합 테스트를 수행.

*   **Step 2: E2E 테스트 자동화:**
    *   `main` 브랜치에 `merge`될 때, `mmp-local-dev` 스택을 `docker-compose`로 실행하고 `pytest -m "e2e"`를 실행하여 완전한 End-to-End 테스트를 수행.

*   **Step 3: 테스트 결과 리포팅:**
    *   `pytest-cov`를 사용하여 테스트 커버리지 리포트를 생성하고, `Codecov`와 같은 서비스에 업로드하여 커버리지 변화를 추적. 