### 목적
- 청사진(`blueprint.md`)과 구현 간 괴리를 해소하고, 런타임 버그/정책/보안을 바로잡으며, 재현성과 테스트 구조를 단단히 하기 위한 “파일 단위·수정 단위” 리팩토링 플랜.

### 핵심 근거(청사진 인용)
- Augmenter 선택 정책과 서빙 금지 요건:
```148:155:blueprint.md
* **Augmenter 선택 정책(요약)**
  - 입력 신호: `environment.app_env`, `feature_store.provider`, `serving.enabled`, `run_mode(train|batch|serving)`, `recipe.model.augmenter(및 fallback)`
  - 규칙:
    1) `local` 또는 `provider=none` 또는 `augmenter.type=pass_through` → `PassThroughAugmenter`
    2) `augmenter.type=feature_store` ∧ `provider in {feast, mock}` ∧ 헬스체크 성공 → `FeatureStoreAugmenter`
    3) (train/batch 전용) 2 실패 ∧ 레시피에 `fallback.sql` 명시 → `SqlJoinAugmenter`
    4) serving에서는 1·3 금지(진입 차단). 실패 시 명확한 에러 반환
```
- Augmenter→Preprocessor 순서:
```203:207:blueprint.md
* **정합성/정책**
  - 서빙은 Feature Store 기반 Augmenter만 허용(폴백/패스스루 금지)
  - 학습/배치는 레시피에 **명시된 경우에 한해** SQL 폴백 허용(자동 폴백 없음)
  - Preprocessor는 Augmenter 이후에 적용되어 훈련-서빙 일관성 유지
```
- 추론 파이프라인 핵심:
```125:134:blueprint.md
4. **예측 호출**: `model.predict(... params={run_mode: "serving"|"batch"})`
   - serving: 온라인 Feature Store 조회(PIT), 폴백/패스스루 금지
   - batch: 오프라인 Feature Store 기본, 레시피 명시 시 SQL 폴백 허용
서버 시작 시 `pass_through`/`sql_fallback` 서빙 진입 차단
```
- 보안/신뢰성:
```257:264:blueprint.md
- 안전한 템플릿/쿼리: allowlist, 금칙어 차단, 타임아웃, LIMIT 가드
- FS 헬스체크, PIT 검증, 운영 로그, 직렬화 안정성
```
- 테스트 원칙:
```267:277:blueprint.md
- Settings 로더/스키마, Factory/Registry(Augmenter 정책), Augmenter 단위,
- 파이프라인 E2E, Serving API 차단, MLflow 스냅샷/시그니처 저장
```

### 리팩토링/개발 플랜(파일별 “무엇을 어떻게”)

#### 1) 치명적 런타임 버그 제거

- `src/engine/factory.py`
  - `create_data_adapter(self)` → `create_data_adapter(self, adapter_type: str | None = None)`로 변경. 기본은 레시피의 `model.loader.adapter` 사용, 명시 인자 우선. `tests/pipelines/train_pipeline.py` 호출과 합치.
  - `create_evaluator`: `evaluator_type` 미정의 버그 수정 → `return EvaluatorRegistry.create(task_type, settings=self.settings)`.
  - `create_augmenter` 전면 재구현(정책 반영):
    - 입력: `run_mode`(옵션), 내부에서 `settings.environment.app_env`, `settings.feature_store.provider in {"none","feast","mock"}`, `settings.serving.enabled`, `recipe.model.augmenter` 판단.
    - 분기: 청사진 3.1 정책(위 인용) 충실 구현. 헬스체크 실패 시 train/batch는 레시피 명시된 `sql_fallback`만 허용, serving은 즉시 차단.
    - 생성 시그니처 정합화: Feature Store Augmenter는 `(settings, factory)` 주입.
  - `create_feature_store_adapter`: 헬스체크 수행(FeastAdapter 초기화 실패 시 명확 에러).
  - `create_pyfunc_wrapper`: 현 로직 유지(Preprocessor→Model→Signature/Schema 추출 OK).

- `src/components/_augmenter/`
  - `__init__.py` 추가: `from ._augmenter import FeatureStoreAugmenter`(기존 `Augmenter`를 `FeatureStoreAugmenter`로 명명 명확화), `from ._pass_through import PassThroughAugmenter`.
  - `src/components/_augmenter/_augmenter.py`
    - 클래스명 `Augmenter` → `FeatureStoreAugmenter`로 변경(의도 명시).
    - 생성자 `(settings, factory)` 유지. `augment(df, run_mode)`에서:
      - run_mode="train"/"batch" → `FeastAdapter.get_historical_features_with_validation(...)`
      - run_mode="serving" → `FeastAdapter.get_online_features(...)`
    - Entity+Timestamp 검증 파라미터 구성(`data_interface_config`)은 Factory가 이미 `PyfuncWrapper`에 포함하므로 사용 가능.
  - `src/components/_augmenter/_sql_fallback.py` 신설:
    - Loader `entity_df` 기준 Left Join 수행. `templating_utils.render_template_from_file/string` 사용.
    - `SqlAdapter.read(sql)`로 피처 조회 후 키(`entity + timestamp`) 기반 merge(left).
    - run_mode가 train/batch일 때만 사용 가능. serving 시 사용 시도 시 예외.
  - `AugmenterRegistry` 구현 위치 명확화:
    - `src/engine/_registry.py`에 `AugmenterRegistry` 추가하거나, `Factory.create_augmenter`가 타입→클래스 직접 매핑(dict) 사용. 권고: 엔진 레벨에 `AugmenterRegistry` 추가하여 `"feature_store" | "pass_through" | "sql_fallback"` 등록.

- `src/components/_preprocessor/_preprocessor.py`
  - `transform`에서 학습 여부 체크 들여쓰기/조건 수정(이미 수정된 형태 확인됨). 예외 메시지 유지.
  - 레시피 스키마 정합성: 현재 구현은 `column_transforms` 기반이므로 스키마도 해당 키 경로를 1급 지원. `steps`는 옵션으로 유지하되 현재 코어 경로는 `column_transforms`.
  - 빈 DF/전열 NaN/타입 혼합 케이스 대비: `fit` 시 ColumnTransformer가 빈 변환 목록이면 경고 후 no-op 파이프라인 구성.

- `src/pipelines/train_pipeline.py`
  - 어댑터 생성 호출부를 `factory.create_data_adapter()`로 교체(레시피 기반) 또는 `create_data_adapter(settings.data_adapters.default_loader)`를 유지하려면 Factory 인자 수용(위에서 수용).
  - 나머지 흐름 유지(MLflow 시그니처/스키마 저장 OK).

- `src/pipelines/inference_pipeline.py`
  - 미정의 함수 교정:
    - `_is_jinja_template(sql)` 간단 구현(`'{{' in sql and '}}' in sql`).
    - 렌더 함수 호출을 `templating_utils.render_template_from_string`로 수정(함수명 일치).
  - 결과 저장 완결:
    - `storage_adapter = factory.create_data_adapter("storage")`
    - `target_uri = settings.artifact_stores["prediction_results"].base_uri + f"/preds_{run.info.run_id}.parquet"`
    - `storage_adapter.write(predictions_df, target_uri)`
  - run_mode="batch"로 `model.predict` 호출.

- `src/serving/router.py`
  - pass_through 외에 `SqlFallbackAugmenter`도 서빙 차단:
    - `if isinstance(wrapped_model.trained_augmenter, (PassThroughAugmenter, SqlFallbackAugmenter)): raise TypeError(...)`
  - 온라인 조회 강제: 필요 시 `wrapped_model.trained_augmenter`가 FeatureStoreAugmenter인지 확인하고, 온라인 엔드포인트 접근 가능성 헬스체크가 실패하면 기동 중단.

#### 2) 정책/보안/청사진 정합화

- `src/settings/loaders.py`
  - `.sql` 경로 처리 보완:
    - 상대 경로면 프로젝트 루트 기준으로 확정 경로로 변환 후 존재성 검사.
    - 존재하면 `prevent_select_star` 호출(SELECT * 차단). 존재하지 않으면 명확한 에러.
  - Jinja 변수를 recipe 명세(`jinja_variables`) allowlist로 검증 후 렌더.

- `src/utils/adapters/sql_adapter.py`
  - 실행 전 보안 가드:
    - `prevent_select_star(sql_query)`
    - 금칙어 검사(DML/DDL 키워드), LIMIT 가드(대용량 보호)
  - 엔진 타임아웃/statement timeout 적용(가능한 DB별 옵션 처리 주석과 함께).
  - 에러 메시지 개선(쿼리 앞 200자 포함).

- `src/utils/system/templating_utils.py`
  - 함수명 표준화: `render_template_from_string`/`render_template_from_file`를 공식명으로 유지. 호출부 정합화.
  - `_validate_context_params` allowlist 강화(청사진 보안 규정 부합).

#### 3) 재현성/튜닝

- `src/utils/system/reproducibility.py` 신설
  - `set_global_seeds(seed: int)`에서 `random`, `numpy`, `torch(옵션)`, `sklearn` 시드 설정.
- `src/pipelines/train_pipeline.py`, `src/pipelines/inference_pipeline.py`
  - 시작 시 `set_global_seeds(settings.recipe.model.computed.get("seed", 42))` 호출 및 로그.
- `src/utils/integrations/optuna_integration.py`
  - `OptunaIntegration` 클래스 구현:
    - 생성자에 `tuning_config`, `seed`, `timeout`, `n_jobs`, pruning 설정.
    - `optimize(objective)`에서 타임박스, 실패/중단 로깅, best_params/best_score/total_trials/optimization_history 반환.
  - Factory에서 `create_optuna_integration`가 실제 클래스를 반환하도록 수정.

#### 4) 서빙 스키마/UX

- `src/serving/schemas.py`
  - `PredictionResponse` 일반화:
    - 기본 필드 `prediction: Any`, `model_uri: str`
    - uplift/특수 태스크는 선택 필드(옵셔널)로 확장.
  - Dynamic Request Model:
    - 현재 Jinja 변수 기반에서 `entity_schema` 기반으로 전환 권장. 단, 현 구현은 loader SQL snapshot에서 PK 파싱을 하므로, 단계적으로 `PyfuncWrapper`의 `entity_schema_snapshot` 혹은 `data_schema` 활용으로 개선(차후 단계).

- `src/serving/_lifespan.py`
  - 동적 입력 스키마 생성 시 `parse_select_columns` 대신, 가능하다면 `wrapped_model.data_schema['entity_columns']` 우선 사용(백필: 기존 방식 fallback).

#### 5) 도커/문서 일관화

- Python 버전 통일: 3.11 권장
  - `pyproject.toml`: `requires-python = ">=3.11,<3.12"`
  - `README.md` 뱃지 “3.11+”로 수정
  - `src/utils/system/environment_check.py`: 3.11 권장/3.12 경고 유지
  - `Dockerfile`: `FROM python:3.11-slim`
- Dockerfile 정리
  - uv.lock 사용: `COPY uv.lock requirements-dev.lock ./` → `uv pip sync --python /opt/venv/bin/python uv.lock ...`
  - `COPY --chown=app:app recipes/ recipes/` (오타 수정)
  - serve 엔트리포인트: `ENTRYPOINT ["/opt/venv/bin/python", "main.py", "serve-api"]`는 동일, `CMD ["--run-id", "YOUR_RUN_ID"]`로 인자 일치
- docker-compose.yml
  - 존재하는 이미지로 교체 또는 `docker/mlflow/Dockerfile` 추가. 로컬 MLflow 포트/볼륨 일치 확인.

#### 6) 테스트 구조(Unit/Integration) 재설계

- `pytest.ini`
  - markers 추가: `unit`, `integration`
- 디렉토리 재배치
  - `tests/unit/` 신설: settings/engine/components/utils/serving 단위 테스트
  - `tests/integration/` 유지: pipelines/serving/feature_store/contracts/infra
- 신규/보강 테스트(요지)
  - Settings: 로더 에러·Jinja·SELECT * 차단
  - Factory: Augmenter 정책 파라미터라이즈(env/provider/run_mode/fallback), Evaluator 생성
  - Augmenter: PassThrough 동일성, FeatureStore 오프라인/온라인 헬스체크 실패/성공, SqlFallback Left Join 키 정합성
  - Inference: 템플릿 렌더 성공/정적 SQL+context_params 실패/결과 저장
  - Serving: pass_through/sql_fallback 차단, 동적 스키마 생성
  - 재현성: 고정 시드로 동일 예측 재현, pip freeze 캡처 여부
  - MLflow: 시그니처/데이터 스키마/스냅샷 저장 유무

### 구현 순서(권장)
1) Factory/Augmenter 시그니처·정책·등록(Registry) 정비
2) Preprocessor 안정화 및 Settings 로더 SQL 검증 보완
3) Inference 파이프라인 완결(템플릿/저장)
4) Serving 정책 보강 및 스키마 일반화
5) 재현성/Optuna 통합
6) Docker/문서 버전 통일
7) 테스트 재구성 + 최소 신규 유닛/통합 추가 → CI에서 unit/integration 분리 실행

### 완료 기준(수용 테스트)
- local(dev off)에서 PassThrough 이외 서빙 기동 차단 확인, dev/prod에서 FS 헬스체크 실패 시 명확 에러 및 train/batch에서만 SQL 폴백 동작
- train/batch/serving 모두 Augmenter→Preprocessor→Model 순서 일관
- 정적 SQL에 SELECT * 포함 시 로딩 단계에서 차단 로그/예외
- 배치 추론 결과가 `artifact_stores.prediction_results`에 저장
- 동일 레시피+시드로 동일 예측 재현 확인
- CI: unit 전부 green, integration 주요 시나리오 green

- 중요한 변경 파일 목록
  - `src/engine/factory.py`(핵심 정책·시그니처)
  - `src/engine/_registry.py`(AugmenterRegistry 추가)
  - `src/components/_augmenter/{__init__.py,_augmenter.py,_sql_fallback.py,_pass_through.py}`
  - `src/components/_preprocessor/_preprocessor.py`
  - `src/settings/{loaders.py,_recipe_schema.py}`(필요 시 경로 처리/유효성 강화)
  - `src/utils/adapters/sql_adapter.py`, `src/utils/system/templating_utils.py`
  - `src/pipelines/{train_pipeline.py,inference_pipeline.py}`
  - `src/serving/{router.py,_lifespan.py,schemas.py}`
  - `src/utils/integrations/optuna_integration.py`(클래스화)
  - 도커/문서: `Dockerfile`, `docker-compose.yml`, `pyproject.toml`, `README.md`
  - 테스트: `tests/unit/**`, `tests/integration/**`, `pytest.ini`

- 청사진 매핑 요약
  - Augmenter 정책/서빙 차단: 3.1, 2.3 인용 라인 적용
  - Augmenter→Preprocessor 순서: 3.6(라인 206)
  - 추론 플로우: 2.3, 8.2 Mermaid
  - 보안/검증: 5절(라인 257-264)
  - 테스트 원칙: 6절(라인 267-277)

작업 착수 가능. 위 순서대로 적용 후, unit→integration 단계별 녹색 확인까지 진행하겠습니다.