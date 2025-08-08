### 목적
- `blueprint.md`와 `next_step.md`를 기준으로 6개 단계별 상세 개발 계획을 확정하고, 실제 구현에 바로 착수할 수 있도록 파일/함수 단위 수정 지침, 인자 호환성, Import 변경점, 시스템 맥락, 검토 체크리스트를 정리한다.

---

#### 1) 치명적 런타임 버그 제거

- 근거(blueprint.md)
  - Augmenter 정책 및 서빙 금지 요건: 148-155
  - Augmenter→Preprocessor 순서: 203-207

- 관련 파일(열람 대상)
  - `src/engine/factory.py`
  - `src/engine/_registry.py`
  - `src/components/_augmenter/_augmenter.py`
  - `src/components/_augmenter/_pass_through.py`
  - `src/components/_preprocessor/_preprocessor.py`
  - `src/pipelines/train_pipeline.py`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) Factory 시그니처/버그 수정
     - `create_data_adapter(self)` → `create_data_adapter(self, adapter_type: str | None = None)`로 변경
       - 우선순위: `adapter_type` 인자 > 레시피 `settings.recipe.model.loader.adapter`
       - 기존 호출부인 `train_pipeline`의 `create_data_adapter(settings.data_adapters.default_loader)`와 레시피 기반 호출 모두 허용
     - `create_evaluator()`에서 미정의 변수 수정: `return EvaluatorRegistry.create(task_type, settings=self.settings)`
  2) Augmenter 패키지/클래스 정비
     - `src/components/_augmenter/__init__.py` 신설: `FeatureStoreAugmenter`, `PassThroughAugmenter` 노출
     - `src/components/_augmenter/_augmenter.py`
       - 클래스명 `Augmenter` → `FeatureStoreAugmenter`로 명확화
       - 생성자 시그니처 유지: `(settings, factory)`
       - `augment(df, run_mode)`에서 run_mode별 분기만 남기고, 실제 FS 조회는 Adapter에 위임
     - `src/components/_augmenter/_pass_through.py`
       - `augment(self, df)` 메서드 구현(현재 `_augment`만 존재) → `BaseAugmenter`의 추상 인터페이스 충족
  3) Augmenter 생성 정책(Factory)
     - 입력 신호: `settings.environment.app_env`, `settings.feature_store.provider in {"none","feast","mock"}`, `settings.serving.enabled`, `recipe.model.augmenter`, `run_mode`
     - 정책(blueprint 148-155):
       - local 또는 provider=none 또는 `augmenter.type=pass_through` → PassThrough
       - `augmenter.type=feature_store` ∧ provider∈{feast,mock} ∧ 헬스체크 성공 → FeatureStore
       - (train/batch 한정) 위 실패 ∧ 레시피에 `fallback.sql` 명시 → SqlFallback
       - serving에서는 PassThrough/SqlFallback 금지(진입 차단)
     - 구현: `AugmenterRegistry` 사용 또는 Factory 내 dict 매핑(권장: Registry)
  4) Registry 확장
     - `src/engine/_registry.py`에 `AugmenterRegistry` 추가: `register(type, cls)`, `create(type, **kwargs)`
     - 부트 시점 등록: `feature_store`, `pass_through`, `sql_fallback`
  5) Train Pipeline 호출 정합화
     - `train_pipeline.py`의 어댑터 생성은 `create_data_adapter()`(레시피 기반) 또는 현 호출 유지(Factory가 인자 수용하므로 둘 다 동작)

- 검토 체크리스트
  - [ ] Factory 어댑터/평가자/증강기 생성 모두 예외 없이 동작
  - [ ] PassThroughAugmenter가 `augment` 구현(추상 메서드 충족)
  - [ ] FeatureStoreAugmenter 생성자에 `factory` 주입됨
  - [ ] Evaluator 생성 시 task_type 전달
  - [ ] 기존 테스트 실행 시 즉시 발생하던 NameError/TypeError 제거

---

#### 2) 정책/보안/청사진 정합화

- 근거(blueprint.md)
  - 보안/신뢰성: 257-264
  - 테스트 원칙: 267-277

- 관련 파일(열람 대상)
  - `src/settings/loaders.py`
  - `src/utils/adapters/sql_adapter.py`
  - `src/utils/system/templating_utils.py`
  - `src/pipelines/inference_pipeline.py`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) Settings 로더 SQL 검증 보완
     - `.sql` 경로가 상대경로이면 프로젝트 루트 기준 절대경로로 변환 후 존재 검사
     - 존재 시 `prevent_select_star` 실행, 미존재 시 명확한 `FileNotFoundError`
     - Jinja 컨텍스트는 레시피의 `jinja_variables` allowlist로 검증 후 렌더
  2) SQL 어댑터 보안 가드
     - 쿼리 실행 전: `prevent_select_star` + 금칙어(DDL/DML) + LIMIT 가드
     - DB별 타임아웃 옵션 주석/적용(가능 시 statement timeout)
     - 에러 메시지에 쿼리 앞 200자 포함(디버깅 용이)
  3) 템플릿 유틸 함수명 표준화
     - 공식 함수명: `render_template_from_string`, `render_template_from_file`
     - 호출부(`inference_pipeline.py`) 함수명 정합화

- 검토 체크리스트
  - [ ] 정적 SQL의 `SELECT *`가 로딩 단계에서 차단
  - [ ] 금칙어/타임아웃/LIMIT 가드 동작
  - [ ] 템플릿 렌더 함수명 불일치 제거
  - [ ] 에러 메시지가 맥락 포함(파일/쿼리 요약)

---

#### 3) 재현성/튜닝

- 근거(blueprint.md)
  - 완전한 재현성: 168-180, 121-134(예측 경로 일관성)

- 관련 파일(열람 대상)
  - `src/utils/system/reproducibility.py`(신설)
  - `src/pipelines/train_pipeline.py`
  - `src/pipelines/inference_pipeline.py`
  - `src/utils/integrations/optuna_integration.py`
  - `src/engine/factory.py`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) 전역 시드 설정 유틸 추가
     - `set_global_seeds(seed: int)`에 `random`, `numpy`, `torch(옵션)`, `sklearn` 시드 설정
     - `train_pipeline.py`/`inference_pipeline.py` 시작부에서 호출, 기본 42 또는 레시피 `computed.seed`
  2) Optuna 통합 클래스화
     - `OptunaIntegration` 구현: `__init__(tuning_config, seed, timeout, n_jobs, pruning)`
     - `optimize(objective)`에서 타임박스/중단/로깅, `best_params/best_score/total_trials/history` 반환
     - Factory `create_optuna_integration()`가 실제 클래스를 반환하도록 수정

- 검토 체크리스트
  - [ ] 동일 레시피+시드로 동일 결과 재현(E2E 허용 오차 내)
  - [ ] Optuna 실패/중단 시 명확 로깅/반환 구조 일관

---

#### 4) 서빙 스키마/UX

- 근거(blueprint.md)
  - 2.3 서빙 플로우/차단 요건: 125-135

- 관련 파일(열람 대상)
  - `src/serving/router.py`
  - `src/serving/_lifespan.py`
  - `src/serving/schemas.py`
  - `src/serving/_endpoints.py`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) 서빙 차단 정책 강화
     - Router 기동 시 `PassThroughAugmenter`뿐 아니라 `SqlFallbackAugmenter`도 차단
     - 온라인 FS 조회 가능성(헬스체크) 실패 시 기동 중단
  2) 입력/응답 스키마 정리
     - `PredictionResponse` 일반화: 기본 `prediction`, `model_uri` + 태스크별 선택 필드(옵셔널)
     - 동적 입력 모델 생성은 가능하면 `wrapped_model.data_schema['entity_columns']` 기반, 불가 시 기존 `parse_select_columns`로 폴백

- 검토 체크리스트
  - [ ] pass_through/sql_fallback 서빙 차단
  - [ ] 온라인 조회 강제 및 실패 시 명확 에러
  - [ ] 응답 스키마가 일반 분류/회귀 시나리오와 호환

---

#### 5) 도커/문서 일관화

- 근거(blueprint.md)
  - 실행 환경/이식성: 168-173, 216-226

- 관련 파일(열람 대상)
  - `pyproject.toml`
  - `README.md`
  - `Dockerfile`
  - `docker-compose.yml`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) Python 버전 통일(3.11 권장)
     - `pyproject.toml` → `requires-python = ">=3.11,<3.12"`
     - `README.md` 배지 3.11+
     - `Dockerfile` 베이스 이미지를 `python:3.11-slim`으로 변경
  2) uv 기반 의존성 설치로 통일
     - `Dockerfile`에서 `uv.lock`/`requirements-dev.lock` 사용
     - 잘못된 경로/디렉토리 교정: `recipes/`(오타 수정), MLflow 포트/URI와 문서/설정 일치
  3) docker-compose 정합화
     - 존재하는 MLflow 이미지 사용 또는 로컬 도커파일 추가
     - 포트/볼륨/환경변수 값이 `config/*.yaml`과 모순 없도록 조정

- 검토 체크리스트
  - [ ] 로컬/CI에서 동일 Python/의존성 트리에 대해 동일 결과
  - [ ] 컨테이너 빌드/서빙 커맨드가 README와 일치

---

#### 6) 테스트 구조(Unit/Integration) 재설계

- 근거(blueprint.md)
  - 테스트 원칙: 267-277

- 관련 파일(열람 대상)
  - `tests/**`, `pytest.ini`, `tests/conftest.py`

- 확정 수정 계획(인자/Import/맥락 포함)
  1) 마커/구조 재정의
     - `pytest.ini`에 `markers = unit: Unit tests; integration: Integration tests`
     - `tests/unit/**` 신설: settings/engine/components/utils/serving의 순수 로직 테스트 이동
     - `tests/integration/**` 유지: pipelines/serving/feature_store/contracts/infra
  2) 신규/보강 테스트
     - Settings: 로더 에러·Jinja·SELECT * 차단
     - Factory: Augmenter 정책 파라미터라이즈(env/provider/run_mode/fallback), Evaluator 생성
     - Augmenter: PassThrough 동등성, FeatureStore(오프라인/온라인), SqlFallback Left Join 키 정합성
     - Inference: 템플릿 렌더 성공/정적 SQL+context 실패, 결과 저장
     - Serving: 차단 정책/동적 스키마
     - 재현성: 고정 시드 동일 결과, pip freeze 캡처
     - MLflow: 시그니처/데이터 스키마/스냅샷 저장

- 검토 체크리스트
  - [ ] `pytest -m unit` 전부 green, 평균 < 1s/파일
  - [ ] `pytest -m integration` 핵심 시나리오 green, flaky 없음

---

### 단계 진행 규칙(반복)
1) 본 문서의 각 단계 제목을 확인하고, 위 “관련 파일(열람 대상)”을 우선 열람한다.
2) 인자 호환성, Import 변경점, 시스템 맥락을 본 문서의 “확정 수정 계획”에 맞춰 구현한다.
3) 로컬 단위 테스트 실행 → 통합 테스트 실행(필요 시 docker-compose) 순으로 검증한다.
4) 체크리스트 충족 시 다음 단계로 진행한다.


