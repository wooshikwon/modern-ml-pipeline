### 테스트 커버리지 확장 전략(문서·구조 준수형)

본 계획은 `tests/README.md`에 정의된 테스트 아키텍처(컨텍스트 기반, 실제 오브젝트, MLflow file://, UUID 명명, 결정론성)를 엄격히 준수하면서, 단순 커버리지 수치 채우기가 아닌 “실제 파이프라인 리스크를 줄이는” 고신뢰 테스트를 추가해 커버리지를 점진적으로 확장하기 위한 실행 문서입니다.

---

### 목표와 원칙
- 목표: 전체 커버리지를 54% → 70%(Phase 1) → 80%(Phase 2)로 단계 상승, 회귀 방지력 강화
- 원칙:
  - 실제 퍼블릭 API만 호출(엔진/내부 재현 금지)
  - 컨텍스트/픽스처 재사용(`mlflow_test_context`, `component_test_context`, `database_test_context`, `make_fail_fast_mlflow`)
  - MLflow 로컬 파일 스토어(`file://{temp}/mlruns`), UUID 접미사 명명
  - 빠른-실패 네트워크 검증 시 공용 헬퍼 사용(중복 금지)
  - 테스트는 작고 독립적, 30초 이내 유지(개별)

---

### 우선 타겟(저커버·핵심 경로)
- 파이프라인/서빙 경로
  - `src/pipelines/inference_pipeline.py` (19%)
  - `src/serving/_endpoints.py` (16%), `src/serving/router.py` (32%), `src/serving/_lifespan.py` (32%)
- CLI/엔트리 포인트·도구
  - `src/__main__.py` (0%), `src/cli/commands/serve_command.py` (31%)
  - `src/cli/utils/config_loader.py` (18%), `src/cli/utils/system_checker.py` (14%)
- 전처리·모델 래퍼·튜너
  - `src/components/preprocessor/modules/encoder.py` (0%), `src/components/preprocessor/preprocessor.py` (15%)
  - `src/models/custom/timeseries_wrappers.py` (19%)
  - `src/components/trainer/modules/optimizer.py` (30%)
- 시스템 유틸
  - `src/utils/system/dependencies.py` (11%), `environment_check.py` (17%), `schema_utils.py` (18%), `templating_utils.py` (20%)

---

### 공통 테스트 패턴(컨텍스트·정책 준수)
- MLflow: `file://{temp_dir}/mlruns`, `uuid4().hex[:8]` 접미사, `MlflowClient.search_runs(...)`
- 데이터/경로: temp 디렉토리 격리, 데이터는 픽스처/컨텍스트가 생성
- 결정론성: 고정 시드 사용(예: 42)
- 네트워크 실패: `make_fail_fast_mlflow()` 호출로 즉시 실패·무재시도

---

### 테스트 시나리오 제안(파일군별)

#### 1) Inference Pipeline (`src/pipelines/inference_pipeline.py`)
- 케이스 A: Happy path – 직전 학습 run의 모델로 배치 추론 성공
  - 준비: `mlflow_test_context.for_classification(...)`로 학습 1회 수행 → 유효 `run_id` 획득
  - 실행: `run_inference_pipeline(settings, run_id, data_path)`
  - 검증: 실행 종료, 아티팩트/로그 최소 존재(실패·예외 없음)
- 케이스 B: 존재하지 않는 `run_id` – 의미있는 실패
  - 준비: `file://` 스토어, 랜덤 `invalid_run_id`
  - 실행/검증: 예외 메시지에 `Run ... not found` 포함
- 케이스 C: 스키마 불일치 – 입력 칼럼 누락/추가 시 방어 동작
  - 준비: 학습 시그니처와 다른 컬럼 구성 CSV
  - 검증: 예외 유형/메시지에 시그니처-입력 불일치 단서

예시 스니펫(개요):
```python
def test_inference_happy_then_missing_run(mlflow_test_context, isolated_temp_directory):
    with mlflow_test_context.for_classification("infer_happy") as ctx:
        run = run_train_pipeline(ctx.settings)
        out = run_inference_pipeline(ctx.settings, run.run_id, ctx.data_path)
        assert out is not None
    bad = run_inference_pipeline(ctx.settings, "nonexistent_run_id_12345", ctx.data_path)
    # 예외 또는 실패 로그 단언
```

#### 2) Serving (`src/serving/_endpoints.py`, `router.py`, `_lifespan.py`)
- 케이스 A: 앱 기동/종료 수명주기 – 에러 없이 start/stop
- 케이스 B: 헬스체크/루트 엔드포인트 200 응답
- 케이스 C: 추론 엔드포인트 – 최소 페이로드로 200 또는 유의미한 4xx
  - 전략: FastAPI `TestClient` 사용, 모델 로드는 `file://`에서 가장 최근 run 혹은 더미 로딩 경로로 스모크
  - 주의: 네트워크 의존 금지, temp 내 파일만 사용

예시 스니펫(개요):
```python
from fastapi.testclient import TestClient
from src.serving.router import app

def test_serving_health_and_predict(isolated_temp_directory):
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    resp = client.post("/predict", json={"inputs": [{"feature_0": 1, "feature_1": 0}]})
    assert resp.status_code in (200, 422)  # 스키마/로드 상태에 따라 의미있는 응답
```

#### 3) CLI/엔트리 포인트 (`src/__main__.py`, `serve_command.py`, `config_loader.py`, `system_checker.py`)
- 케이스 A: `python -m src --help` 출력(비상태 스모크)
- 케이스 B: `serve_command` – 필수 옵션 누락/잘못된 포트 시 명확한 에러
- 케이스 C: `config_loader` – 최소 유효/무효 YAML 로드 분기(검증 실패 메시지 확인)
- 케이스 D: `system_checker` – 필수 의존성 미설치 시 경고/보고 포맷 확인

예시 스니펫(개요):
```python
from typer.testing import CliRunner
from src.cli.main_commands import app

def test_cli_help():
    result = CliRunner().invoke(app, ["--help"]) 
    assert result.exit_code == 0

def test_config_loader_invalid_yaml(tmp_path):
    cfg = tmp_path/"bad.yaml"; cfg.write_text("invalid: [yaml")
    # load_settings 호출 경로를 통해 실패 단언
```

#### 4) Preprocessor (`encoder.py`, `preprocessor.py`)
- 케이스 A: 카테고리 인코더 – 학습·추론 시 미지의 카테고리 처리
- 케이스 B: 수치 스케일러+결측치 대치 – 파이프 조합의 round-trip 통과
- 케이스 C: `Preprocessor` 오케스트레이션 – 선택적 단계 on/off 분기

예시 스니펫(개요):
```python
def test_encoder_handles_unseen_categories():
    df_train = pd.DataFrame({"cat": ["a","b"], "target":[0,1]})
    df_test = pd.DataFrame({"cat": ["c"]})
    # encoder.fit(df_train) → transform(df_test) 예외 없이 완료
```

#### 5) Trainer Optimizer (`src/components/trainer/modules/optimizer.py`)
- 케이스 A: HPO off – 고정 하이퍼파라미터 경로로 훈련 1회
- 케이스 B: HPO on(optuna) – 목적함수 최소 호출 보장(스몰 스페이스)
- 케이스 C: 알 수 없는 옵티마이저 키 – 레지스트리 조회 실패 메시지 확인

#### 6) Timeseries Wrappers (`src/models/custom/timeseries_wrappers.py`)
- 케이스 A: 윈도우 길이·특성 수 불일치 시 명확한 예외
- 케이스 B: 최소 길이 시퀀스에서 단일 스텝 예측 성공

#### 7) System Utils (`dependencies.py`, `environment_check.py`, `schema_utils.py`, `templating_utils.py`)
- 케이스 A: 오프라인 환경에서 필수/선택 의존성 감지 결과 포맷 검증
- 케이스 B: 환경 점검 리포트 JSON 스키마 키 존재
- 케이스 C: DataFrame → 스키마 유틸 최소 변환 경로(숫자/문자열/날짜)
- 케이스 D: Jinja 템플릿 최소 렌더링(변수 바인딩/빈 변수 경고)

---

### 작성 가이드(짧은 체크리스트)
- 컨텍스트·픽스처 우선: 설정·자원 준비는 컨텍스트로 이동
- MLflow 정책 준수: `file://`, `uuid`, `search_runs` 사용
- 상태 격리: 각 테스트는 고유 temp 디렉토리 사용, 전역 공유 금지
- 실패 경로는 의미있는 메시지 단언(단순 True 금지)
- 성능: 각 테스트 < 30초, 외부 네트워크/원격 스토어 금지
- **파일 명명 규칙**: Phase1, Phase2, temp, draft 등 임시적 개발 단어 금지, 의미있는 이름 사용

---

### 단계적 실행 계획
- 1단계 (우선순위 높음): 파이프라인/서빙·CLI 스모크 + Inference 필수 케이스
  - 목표 커버리지: ~70%
  - 대상: Inference 3케이스, Serving 3케이스, CLI 3케이스, Config Loader 2케이스
- 2단계 (우선순위 중간): Preprocessor·Trainer Optimizer·Timeseries Wrappers·System Utils 기본 분기
  - 목표 커버리지: ~80%
  - 대상: Encoder/Preprocessor 3케이스, Optimizer 3케이스, TS Wrappers 2케이스, Utils 4케이스
- 3단계 (지속적 개선): 리그레션 추가, 경계/에지 케이스 보강, 플레이키 제거

---

### 수용 기준(Definition of Done)
- 모든 신규 테스트가 `tests/README.md` 정책을 준수
- 실패 메시지 단언 포함(원인 파악 가능)
- 병렬 실행(`-n auto`) 안정, flakiness 없음
- 커버리지 리포트에 파일별 미싱 라인 감소 확인

---

### 운영 메모
- HTML 리포트: `pytest --cov=src --cov-report=html`로 세부 미싱 라인 확인
- 가장 저렴한 커버리지부터: 에러 경로·헬스체크·스모크로 빠르게 리스크 감소
- 공통 헬퍼 재사용: 네트워크 실패, MLflow 컨텍스트, SettingsBuilder

