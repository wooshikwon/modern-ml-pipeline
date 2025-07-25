# Modern ML Pipeline 리팩토링 및 안정화 마스터 플랜 (v2)

## 1. 최종 목표 (The Ultimate Goal)

현재 시스템에서 발견된 아키텍처적 문제들을 해결하여, **단순하고(Simple), 명확하며(Explicit), 확장 가능한(Scalable)** 구조를 확립한다. 이를 통해 개발 생산성을 극대화하고, 어떤 환경에서도 예측 가능하게 동작하는 견고한 ML 시스템을 완성하며, 최종적으로 **외부에 공개 가능한 라이브러리**로 만든다.

이 문서는 그 목표를 달성하기 위한 **구체적이고 실천적인 실행 계획(Actionable Steps)** 이다.

## 2. 전체 로드맵 (The Grand Roadmap)

개발은 아래 4개의 논리적 단계에 따라 순차적으로 진행한다. 각 단계는 이전 단계의 성공적인 완료를 전제로 한다.

| 단계 | 목표 | 핵심 기술/전략 | **핵심 결과물** | 예상 기간 |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: 기반 재설계** | 시스템의 뼈대를 바로잡고 복잡성을 제거한다. | 3계층 아키텍처, 명시적 Registry | `src/core` 완전 제거, 단순화된 `Registry`와 `Factory` | 3-5일 |
| **Phase 2: 어댑터 현대화** | 외부 시스템 I/O를 표준화하고 유연성을 극대화한다. | Feast Native, fsspec, SQLAlchemy | 통합 `Sql/Storage/Feast` 어댑터, 레거시 어댑터 완전 제거 | 4-6일 |
| **Phase 3: CLI 및 설정 파이프라인 구축** | 라이브러리로서의 사용성을 완성한다. | Typer CLI, Jinja2, 설정 빌더 | `init`, `validate` 등 신규 CLI, Jinja 템플릿 지원 | 3-4일 |
| **Phase 4: 최종 통합 및 검증** | 리팩토링된 시스템의 안정성과 완성도를 보장한다. | E2E 테스트, 문서화 | 리팩토링된 시스템의 E2E 테스트 통과, 현대화된 사용자 문서 | 2-3일 |

```mermaid
graph TD
    subgraph "Phase 1: 기반 재설계 (Foundation)"
        A[Step 1.1: 3계층 아키텍처 확립] --> B[Step 1.2: Factory 역할 단순화]
    end
    
    subgraph "Phase 2: 어댑터 현대화 (Modernization)"
        B --> C[Step 2.1: 통합 SQL/Storage/Feast 어댑터 구현]
        C --> D[Step 2.2: Registry 업데이트 및 레거시 어댑터 제거]
    end

    subgraph "Phase 3: CLI 및 설정 파이프라인 구축 (Interface)"
        D --> E[Step 3.1: Config 파일 구조 개편]
        E --> F[Step 3.2: '설정 빌더' 파이프라인 구현 (Jinja 포함)]
        F --> G[Step 3.3: 신규 CLI 인터페이스 구현]
    end

    subgraph "Phase 4: 최종 통합 및 검증 (Validation)"
        G --> H[Step 4.1: PyfuncWrapper 저장 로직 검증]
        H --> I[Step 4.2: End-to-End 통합 테스트]
        I --> J[Step 4.3: 사용자 문서 현대화]
    end

    style A fill:#D5F5E3,stroke:#2ECC71
    style B fill:#D5F5E3,stroke:#2ECC71
    style C fill:#EBF5FB,stroke:#3498DB
    style D fill:#EBF5FB,stroke:#3498DB
    style E fill:#FEF9E7,stroke:#F1C40F
    style F fill:#FEF9E7,stroke:#F1C40F
    style G fill:#FEF9E7,stroke:#F1C40F
    style H fill:#FDEDEC,stroke:#E74C3C
    style I fill:#FDEDEC,stroke:#E74C3C
    style J fill:#FDEDEC,stroke:#E74C3C
```

---

## 3. 상세 실행 계획 (라이브러리화 최종안)

### **Phase 1: 기반 재설계 (Foundation Refactoring)**

**목표:** 라이브러리화에 앞서, 시스템 내부의 구조적 문제를 해결하고 견고한 기반을 다진다.

*   **Step 1.1: 3계층 아키텍처 확립 및 Import 경로 수정**
    *   **작업:** `grep`을 사용하여 프로젝트 전체에서 `from src.core` 구문을 검색하고, 이를 새로운 3계층 구조(`components`, `engine`, `pipelines`)에 맞게 모두 수정한다.
    *   **완료 조건:** `src/core` 디렉토리가 완전히 삭제되고, 모든 `import`문이 수정되어 `pytest` 기본 테스트가 정상적으로 실행된다.

*   **Step 1.2: `Factory` 역할 단순화**
    *   **작업:** `src/engine/factory.py`를 수정한다. `_get_adapter_class`, `create_data_adapter_legacy` 등 `AdapterRegistry`와 책임이 중복되는 모든 매핑/해석 로직과 레거시 메서드를 과감히 제거한다.
    *   **완료 조건:** `Factory` 클래스에는 오직 `AdapterRegistry.create()`를 호출하여 컴포넌트를 생성하는 로직만 남는다.

### **Phase 2: 어댑터 현대화 및 통합 (Adapter Modernization)**

**목표:** 개별 기술에 종속된 여러 어댑터를 업계 표준 라이브러리를 활용한 단일 통합 어댑터로 교체하여, 복잡성을 극적으로 낮추고 유연성을 극대화한다.

*   **Step 2.1: 통합 `SqlAdapter`, `StorageAdapter`, `FeastAdapter` 구현**
    *   **작업:** `src/utils/adapters/`에 `sql_adapter.py`, `storage_adapter.py`, `feast_adapter.py`를 신규 생성한다.
        - `SqlAdapter`: `SQLAlchemy` 기반으로 동작
        - `StorageAdapter`: `fsspec` 기반으로 동작
        - `FeastAdapter`: `feast` 라이브러리를 직접 사용하는 가벼운 래퍼
*   **Step 2.2: `AdapterRegistry` 업데이트 및 레거시 어댑터 제거**
    *   **작업:** `src/engine/registry.py`의 `register_all_adapters` 함수를 수정하여, 새로 만든 `SqlAdapter`, `StorageAdapter`, `FeastAdapter`를 각각 `"sql"`, `"storage"`, `"feature_store"` 타입으로 명시적으로 등록한다.
    *   **작업:** `_register_legacy_adapters_temporarily` 함수와 `src/utils/adapters/`에 존재하던 모든 개별 어댑터 파일들(`bigquery_adapter.py`, `gcs_adapter.py` 등)을 **모두 삭제**한다.

### **Phase 3: CLI 및 설정 파이프라인 구축 (CLI & Settings Pipeline)**

**목표:** 라이브러리 사용자에게 명확하고 유용한 CLI 인터페이스를 제공하고, Jinja 템플릿을 지원하는 견고한 설정 로딩 파이프라인을 구축한다.

*   **Step 3.1: Config 파일 구조 개편**
    *   **작업:** `config/data_adapters.yaml` 파일을 신규 생성하고, `base.yaml`의 `data_adapters` 섹션을 이곳으로 옮긴다.
    *   **작업:** `config/base.yaml`에서 불필요한 섹션을 정리하고, 핵심적인 인프라 정보만 남도록 단순화한다.
*   **Step 3.2: "설정 빌더" 파이프라인 구현 (Jinja 포함)**
    *   **작업:** `src/utils/system/templating_utils.py` 파일을 신규 생성하고, Jinja2 템플릿 렌더링 함수를 구현한다.
    *   **작업:** `src/settings/loaders.py`를 수정하여 **[YAML 로드 → Jinja 렌더링 → Pydantic 검증]**의 3단계 설정 파이프라인을 구현한다.
*   **Step 3.3: 신규 CLI 인터페이스 구현**
    *   **작업:** `main.py`를 수정하여 `Typer` 기반으로 `train`, `batch-inference`, `serve-api`, `init`, `validate`, `test-contract` 커맨드를 모두 구현한다. `train`과 `batch-inference`는 `--context-params` 옵션을 포함한다.

### **Phase 4: 최종 통합 및 검증 (Finalization & Validation)**

**목표:** 리팩토링된 모든 구성요소가 완벽하게 함께 동작함을 검증하고, 변경사항을 모든 사용자 문서에 반영하여 프로젝트를 마무리한다.

*   **Step 4.1: `PyfuncWrapper` 저장 로직 검증**
    *   **작업:** `src/engine/factory.py`의 `create_pyfunc_wrapper` 함수를 최종 검토하여, `loader_sql_snapshot`에 **Jinja 렌더링이 완료된 최종 SQL 문자열**이 저장되도록 보장한다.
*   **Step 4.2: End-to-End 통합 테스트**
    *   **작업:** 새로운 CLI (`--context-params` 포함)를 사용하여 학습 파이프라인을 실행하고, 생성된 `run-id`를 사용하여 `batch-inference`와 `serve-api`가 모두 정상적으로 동작하는지 전체 흐름을 테스트한다.
*   **Step 4.3: 사용자 문서 현대화**
    *   **작업:** `README.md`와 `docs/DEVELOPER_GUIDE.md`를 수정하여, 새로운 CLI 사용법, `init` 커맨드, 단순화된 `config` 구조, Jinja 템플릿 작성법 등 모든 변경사항을 반영한다.

---

## 4. 리팩토링 완료 후 재개 계획

위의 모든 리팩토링이 완료되고 시스템이 안정화된 후, 아래의 기존 기능 고도화 계획들을 순서대로 재개한다.

- **Feature Store 심층 테스트 재개**
- **성능 및 안정성 테스트**
- **CI/CD 자동화**

---

## 부록 (Appendix)

### A.1: 발견된 문제점 상세

- **🔥 Critical: Import/Registry 패턴 충돌:** `src/core/factory.py`와 `src/utils/adapters/__init__.py`에서 어댑터 클래스를 직접 `import`하여, 데코레이터 기반 Registry 패턴과 혼재되며 `pytest` 실행 시 `namespace package` 충돌 발생.
- **📊 Major: FeatureStoreAdapter 과도한 복잡성:** 369줄의 거대한 단일 클래스가 Redis에 강하게 종속되어, Feast가 지원하는 다른 Online Store(DynamoDB, PostgreSQL 등)로의 확장을 원천 차단.
- **⚙️ Medium: 복잡한 매핑 로직 중복:** 동일한 어댑터 매핑 로직이 Registry, Factory, Legacy 메서드 등 3곳 이상에서 중복되어 유지보수성과 일관성 저해.

### A.2: 기존 리팩토링 계획 (v1 - History)

- #### **Phase R1: Registry 패턴 완전 단순화 (최우선, 2-3일)**
  - **목표**: 복잡한 메타프로그래밍 제거, 단순한 딕셔너리 기반 Registry로 전환
- #### **Phase R2: Factory 단순화 (중요도 높음, 1-2일)**
  - **목표**: 매핑 로직 단순화, 오직 생성(instantiation) 책임만 수행
- #### **Phase R2-A: 3층 아키텍처 구조 개편 (병렬 진행, 1-2일)**
  - **목표**: 현재 혼재된 `src/core/` 구조를 명확한 3층 구조로 재편
- #### **Phase R3: FeatureStore → Feast Native (혁신적 변화, 3-5일)**
  - **목표**: Redis 종속성 완전 제거, Feast의 네이티브 Online Store 지원 활용
- #### **Phase R4: 어댑터 통합 및 추상화 (새로운 단계, 3-4일)**
  - **목표**: `fsspec`과 `SQLAlchemy`를 활용하여 여러 어댑터를 단일 통합 어댑터로 통합
- #### **Phase R5: 통합 테스트 및 검증 (최종, 2-3일)**
  - **목표**: 리팩토링된 전체 구조로 모든 기존 테스트 통과 확인

### A.3: 완료된 작업 (기존 History)

- **Phase 1: 기반 구축 및 계약 확립 (완료)**
- **Phase 2: 테스트 스위트 전체 현대화 (완료)** 