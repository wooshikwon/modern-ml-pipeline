## 테스트 안정화 및 그룹별 메트릭 수집 실행 계획

### 목적
- 전체 테스트를 e2e, integration, unit으로 분할 실행하고, 각 그룹별로 다음 4가지 메트릭을 수집합니다.
  - 커버리지, 스킵율, 실패율, 속도(총 소요시간 및 슬로우 테스트)
- MLflow/uvicorn/Feast/SQLite 등 외부 리소스 사용 테스트의 병렬 충돌과 잔류 프로세스 문제를 제거합니다.

### 요약 진단
- 전역 프로세스 종료 로직이 병렬 워커 간 간섭을 유발합니다.
  - `tests/conftest.py`의 `pytest_sessionstart/finish`, `ensure_deterministic_execution`에서 `pgrep/pkill`로 `mlflow.server`, `uvicorn.*mlflow`를 전역 종료 → 다른 워커 작업을 중간에 끊어 타임아웃/간헐 실패 발생 가능.
- MLflow 트래킹 경로가 테스트 사이에서 공유되어 SQLite 파일 락 충돌이 발생합니다.
  - 기본값 `sqlite:///tests/fixtures/databases/test_mlflow.db` 사용 경로가 섞이면 병렬 시 락 경쟁.
- 서버 라이프사이클이 픽스처로 엄격히 관리되지 않아 잔류/중복 프로세스가 발생할 수 있습니다.
- 그룹별(유닛/통합/E2E)로 분리된 메트릭 수집 파이프라인이 부재합니다.

---

### 메트릭 정의 및 산출 방식
- 커버리지: pytest-cov로 그룹별 커버리지 파일/리포트 생성
  - COVERAGE_FILE=.coverage.<group>, `--cov=src` `--cov-report=xml:coverage.<group>.xml` `--cov-report=term-missing`
- 스킵율/실패율: pytest 요약(`-rA`) 파싱
  - “X passed, Y failed, Z skipped, W deselected”에서 비율 계산
- 속도: 전체 실행 시간과 `--durations=0` 슬로우 테스트 목록 수집

---

### Phase 1 — 전역 종료 로직 옵트인 전환 및 기본 격리 강화
- 목표: 병렬 워커 간 간섭 제거, 안전한 기본값 제공
- 조치:
  - 전역 프로세스 종료 로직을 환경변수 게이트로 보호
    - `tests/conftest.py`의 `cleanup_mlflow_processes()` 호출(세션 시작/종료, 테스트 종료 후 부분)을 `MMP_ENABLE_GLOBAL_KILL=1`일 때만 수행하도록 조건화
    - 기본값은 비활성화(= 전역 pkill 미실행)
  - MLflow 트래킹 기본 경로를 테스트 격리형으로 통일
    - 기본 `SettingsBuilder` 초기값에서 공유 sqlite 파일 사용을 지양
    - 원칙: 가능하면 `file://{tmp_dir}/mlruns` (이미 존재하는 `isolated_mlflow_tracking` 픽스처 사용)으로 테스트 스코프별 격리
    - 공유 DB 경로를 반드시 써야 할 테스트는 `-n 1` 또는 락 처리(Phase 2)

---

### Phase 2 — 서버/외부 리소스 테스트 직렬화 락
- 목표: MLflow/서버/포트/파일 락 충돌 제거
- 조치:
  - 파일 락 기반 세션 스코프 락 픽스처 추가(예: `/tmp/mmp-server.lock`)
  - 서버/리소스에 민감한 테스트에 마커(예: `@pytest.mark.server`)를 부여하고 해당 마커 테스트에서 락 픽스처 사용
  - CI/로컬 실행 시 `-m "not server"`는 병렬 허용, `-m server`는 `-n 1` 또는 락으로 직렬화

---

### Phase 3 — 서버 라이프사이클 픽스처 도입(필요 시)
- 목표: 테스트 스코프에서 시작-준비대기-종료 보장
- 조치:
  - MLflow 서버/임의 포트 선택/헬스체크 폴링/정상 종료를 포함하는 픽스처 제공
  - uvicorn이 필요한 경우에도 동일 패턴 적용(단, 현재 테스트는 FastAPI `TestClient`로 충분히 커버)
  - 모든 서버성 테스트는 전역 pkill에 의존하지 않고 해당 픽스처가 종료 보장

---

### Phase 4 — 그룹 분할 실행과 그룹별 메트릭 산출
- 목표: unit/integration/e2e로 나눠 독립 실행 및 메트릭 분리 수집
- 실행 예시(로컬):
  - Unit
    - `COVERAGE_FILE=.coverage.unit pytest tests/unit -n auto --cov=src --cov-report=xml:coverage.unit.xml --cov-report=term-missing -rA --durations=0`
  - Integration
    - `COVERAGE_FILE=.coverage.integration pytest tests/integration -n 1 --cov=src --cov-report=xml:coverage.integration.xml --cov-report=term-missing -rA --durations=0`
    - 서버/DB/파일락 민감 테스트 포함 시 직렬화 권장
  - E2E
    - `COVERAGE_FILE=.coverage.e2e pytest tests/e2e -n 1 --cov=src --cov-report=xml:coverage.e2e.xml --cov-report=term-missing -rA --durations=0`
- 메트릭 집계
  - 스킵/실패/패스 수: pytest 출력(`-rA`) 파싱 → JSON 저장
  - 속도: 총 경과 시간과 `--durations=0` 목록 파싱 → JSON 저장
  - 커버리지: 각 그룹의 xml을 보존, 필요 시 `coverage combine .coverage.unit .coverage.integration .coverage.e2e && coverage xml -o coverage.total.xml`

---

### Phase 5 — CI 통합 및 아티팩트 관리
- 목표: CI에서 안정적으로 실행/수집/리그레션 확인
- 조치:
  - 워크플로 단계 분리: unit → integration → e2e
  - 각 단계에서 `COVERAGE_FILE`을 다르게 설정하여 커버리지 분리 저장
  - 아티팩트 업로드: `coverage.*.xml`, 그룹별 메트릭 JSON, `htmlcov/*`(옵션)
  - 병렬 정책: unit은 `-n auto`, integration/e2e는 기본 `-n 1`(또는 `@server`만 직렬화)
  - 과거 대비 리그레션 가드: 실패율/스킵율/속도 경계치 설정(예: 실패율 0, 스킵율 < 15%, 총 시간 SLA)

---

### Phase 6 — 검증, 리스크 및 롤백
- 검증 체크리스트
  - 병렬(unit) + 직렬(integration/e2e) 조합에서 타임아웃 미발생
  - 잔류 MLflow/uvicorn 프로세스 없음(전역 pkill 비활성, 픽스처 종료로 해결)
  - 그룹별 커버리지/스킵/실패/속도 산출물 생성 확인
- 리스크/대응
  - 기존 테스트가 기본 MLflow 설정을 가정하는 경우: 해당 테스트를 `isolated_mlflow_tracking` 사용으로 전환
  - 락으로 인한 테스트 지연: `@server` 범위를 최소화하고 나머지는 병렬화 유지
  - 전역 pkill이 반드시 필요한 환경: `MMP_ENABLE_GLOBAL_KILL=1`로 명시적 옵트인

---

### 참고(파일/구현 포인트)
- 전역 종료 로직 위치
  - `tests/conftest.py` → `pytest_sessionstart`, `pytest_sessionfinish`, `ensure_deterministic_execution` 내 `pgrep/pkill`
- MLflow 격리 픽스처
  - `tests/conftest.py` → `isolated_mlflow_tracking` (file://mlruns)
- 서빙 테스트 컨텍스트(서버 미기동, TestClient 사용)
  - `tests/fixtures/contexts/serving_context.py`
  - `tests/integration/test_serving_integration.py`
- uvicorn 실제 기동은 회피(유닛 테스트에서 run 호출만 모킹)
  - `tests/unit/cli/test_serve_command.py`

---

### 권장 실행 요약
- 로컬
  - Unit: 병렬 실행으로 빠르게 피드백, 그룹 커버리지/메트릭 저장
  - Integration/E2E: 직렬 실행(또는 `@server`만 직렬), 그룹 커버리지/메트릭 저장
- CI
  - 3단계 파이프라인으로 분리, 각 단계 산출물을 업로드 및 임계치 검증

이 문서는 테스트 안정화와 그룹별 메트릭 수집을 위한 실천 가능한 단계별 계획을 제공합니다. 위 변경을 적용한 후, 타임아웃/충돌 감소와 함께 그룹별 정확한 품질 지표를 안정적으로 확보할 수 있습니다.