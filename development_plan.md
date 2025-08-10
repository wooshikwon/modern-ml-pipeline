### 목적
- 본 문서는 “Stage 6: 전역 임포트/의존성 안정화 + 테스트 전면 정비” 실행 가이드입니다. 이미 완료된 부분은 제외하고, 남은 항목을 구현 단위로 제시합니다.

---

## 1) 전역 부트스트랩/의존성 검증(6-5)

- 작업
  - `src/utils/system/dependencies.py` 신설
    - `validate_dependencies(settings)` 구현
      - 기능별 필수 패키지 집합 구성 후 import 시도, 실패 시 `ImportError(패키지명)` 즉시 발생
      - 예시 규칙
        - recipe.loader.adapter == storage && parquet 사용 → `pyarrow`
        - recipe.loader.adapter == sql → `sqlalchemy`
        - feature_store.provider == feast → `feast`
        - hyperparameter_tuning.enabled → `optuna`
        - serving.enabled → `fastapi`, `uvicorn`
  - `src/engine/__init__.py::bootstrap(settings)`에서 `register_all_components()` 후 `validate_dependencies(settings)` 호출
- 수용 기준
  - 활성 기능에 필요한 패키지가 없으면 ImportError로 즉시 실패(패키지명 포함)

---

## 2) Factory 엄격화(6-6)

- 작업
  - `src/engine/factory.py`에서 AdapterRegistry가 비어있을 때 내부에서 `register_all_components()`를 호출하는 fallback 제거
  - 부트스트랩 호출 누락 시 명확히 실패하도록 유지
- 수용 기준
  - 부트스트랩 미호출 시 어댑터 생성이 실패하며, 메시지에 가용 어댑터 목록만 노출

---

## 3) 데이터 분할 stratify 가드(6-7)

- 작업
  - `src/components/_trainer/_data_handler.py::split_data`
    - stratify 적용 전 조건 검사 추가
      - 타깃/처치 컬럼 존재 && 최소 각 클래스/그룹 빈도 ≥ 2 && 표본 크기 충분 시에만 stratify 적용
      - 조건 미충족 시 `stratify=None`로 분할
- 수용 기준
  - 소표본/불균형 CSV로도 분할이 안정적으로 수행됨(파이프라인 테스트 green)

---

## 4) Serving 초기화에 부트스트랩 보장(6-8)

- 작업
  - `src/serving/_lifespan.py` 또는 `router.setup_api_context` 초기에 `bootstrap(settings)` 호출 보장
- 수용 기준
  - LOCAL에서 서빙 차단 정책 유지, DEV에서 의존성/레지스트리 만족 시 정상 동작

---

## 5) Import-linter 계약 추가(6-9)

- 작업
  - `pyproject.toml`에 계약 추가(예시)
    - components → engine/settings 상향 의존 금지
    - engine → components 허용(팩토리/레지스트리)
    - serving → pipelines 직접 의존 금지
  - CI에서 계약 위반 시 실패하도록 설정(추가 PR로 분리 가능)
- 수용 기준
  - 계약 위반 없음(기본 규칙으로 시작 후 점진 강화)

---

## 6) 테스트/픽스처 정비(6-10, 6-11)

- 작업
  - 유닛/파이프라인 테스트
    - storage + CSV/Parquet로만 데이터 로딩
    - `tests/fixtures/data/*`에 50+행, 타깃 클래스 최소 10+ 보장하는 데이터셋 추가/대체
    - 파이프라인 테스트는 파일명 규칙/최소 메타만 검증(반환 오브젝트 의존 최소화)
  - SQL/Feast 관련 테스트
    - `@pytest.mark.requires_dev_stack` 마커 부착
    - 유닛에서는 SQL 가드/경로 존재만, 쿼리 실행은 통합 환경에서만
- 수용 기준
  - `tests/pipelines/*` 및 `tests/utils/*` green
  - DEV 스택 미기동 시 `@requires_dev_stack` 자동 스킵

---

## 7) 불필요한 예외/우회 로직 제거(6-12)

- 작업
  - `src/engine/factory.py`의 레지스트리 지연등록 제거(2와 중복이지만 문서상 명시)
  - 테스트에서 임시 sqlite 연결/강제 SQL 경로 설정 제거(유닛은 storage 고정)
  - 과도한 try/except 제거(환경 미준비 시 명시적 실패 유지)
- 수용 기준
  - 우회 없이 선명한 실패/성공 경로. 실패 시 메시지로 원인(패키지/부트스트랩/경로)을 즉시 파악 가능

---

## 8) 실행 순서
1) 의존성 검증 유틸 추가 및 `bootstrap` 연동(6-5)
2) Factory fallback 제거(6-6)
3) 분할 stratify 가드(6-7)
4) 테스트 픽스처 보강 및 파이프라인 테스트 안정화(6-10)
5) 서빙 부트스트랩 보장(6-8)
6) Import-linter 계약 추가(6-9)
7) SQL/Feast 통합 테스트 경계 정리(6-11)
8) 불필요 예외/우회 제거 마무리(6-12)

---

## 9) 최종 수용 기준(전체)
- LOCAL: 유닛+파이프라인 green, 작은 데이터에서도 분할 실패 없음
- 의존성 검증: 미설치 패키지 시 ImportError 즉시 실패(패키지명 포함)
- 부트스트랩 규율: Factory 내부 fallback 제거로 누락 즉시 실패
- Import-linter: 계약 위반 없음
- 픽스처: storage 기반 데이터로 파이프라인 테스트 안정 통과


