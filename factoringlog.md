작업 계획: 단계 1 - 치명적 버그 및 인터페이스 정합성 제거
[PLAN] development_plan.md - 1) 치명적 런타임 버그 및 인터페이스 정합성 제거
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- BaseAugmenter에 run_mode 인자 추가, PassThroughAugmenter augment 구현, FeatureStoreAugmenter로 클래스 명확화
- AugmenterRegistry를 엔진 레벨에 신설하고 Factory가 이를 사용
- Factory: create_data_adapter(adapter_type) 시그니처 확장, create_evaluator NameError 수정, create_augmenter(run_mode) 정책 구현
- Trainer/Optimizer 정합화(최소 변경: best_artifacts 추적), DataHandler 호출 불일치 제거

**** src/interface/base_augmenter.py
(근거) development_plan.md 1) 및 승인된 CoT에 따라 BaseAugmenter 인터페이스에 run_mode 인자 추가.
(상세) augment(self, data, run_mode="batch") 시그니처로 변경, 간단한 Args 설명 추가.
(자가 비평) 인터페이스 변경이 PyfuncWrapper.predict 호출(run_mode 전달)과 정합. 다른 구현체가 컴파일 타임에 누락되지 않도록 PassThrough/FS 구현 동반 수정.

**** src/components/_augmenter/_pass_through.py
(근거) development_plan.md 1) PassThroughAugmenter가 추상 메서드 충족.
(상세) _augment 제거, augment(self, df, run_mode) 구현. 입력 그대로 반환.
(자가 비평) 단순 경로로 예외 없음. 서빙 차단은 Router/Factory에서 처리될 예정.

**** src/components/_augmenter/_augmenter.py
(근거) development_plan.md 1) FeatureStoreAugmenter 명확화 및 run_mode별 분기.
(상세) 클래스명을 FeatureStoreAugmenter로 정리, offline은 get_historical_features_with_validation, online은 get_online_features 사용. entity_rows 변환 포함.
(자가 비평) features 평탄화 규칙은 네임스페이스:피처 형식으로 구성. FeastAdapter API와 정합.

**** src/engine/_registry.py
(근거) development_plan.md 1) AugmenterRegistry 신설.
(상세) AugmenterRegistry 추가, __init__에서 augmenter 모듈 임포트 트리거하여 등록 유도.
(자가 비평) 등록 순서 이슈 대비 try/except 처리. 다음 단계에서 sql_fallback 등록 예정.

**** src/components/_augmenter/__init__.py
(근거) development_plan.md 1) 공개 심볼/등록.
(상세) FeatureStoreAugmenter, PassThroughAugmenter 노출 및 AugmenterRegistry.register 수행.
(자가 비평) 초기 임포트 순서 이슈는 예외 무시로 방어.

**** src/engine/factory.py
(근거) development_plan.md 1) Factory 시그니처/버그/정책 구현.
(상세) create_data_adapter(adapter_type) 추가, create_evaluator NameError 수정, create_augmenter(run_mode) 정책 구현. PassThrough/FS 선택 및 serving 차단 케이스 명확화.
(자가 비평) sql_fallback은 NotImplementedError로 후속 단계 예약. 호출부는 기존과 역호환.

작업 계획: 단계 2 - 정책/보안/청사진 정합화
[PLAN] development_plan.md - 2) 정책/보안/청사진 정합화
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- Settings 로더 import 정리 및 정적 SQL 경로/보안 강화
- Serving 설정에 enabled 추가, 서빙 차단 정책 보강
- Recipe 스키마 검증 교정
- SQL 어댑터 가드 강화, 템플릿 함수명 정합화
- 배치 추론 결과 파일명 `preds_{run_id}.parquet` 고정

**** src/settings/loaders.py
(근거) development_plan.md 2) 정책/보안/청사진 정합화
(상세) Settings import를 `from .schema import Settings`로 통일. 정적 `.sql` 경로 절대화 및 존재 검사, `prevent_select_star` 적용을 강화.
(자가 비평) 청사진 5절 보안 원칙에 부합. 파일 미존재 시 명확한 예외, SELECT * 차단 로직 유지. 기존 호출부와 하위 호환 유지.

**** src/settings/_config_schema.py
(근거) development_plan.md 2) Serving 설정 보강.
(상세) `ServingSettings`에 `enabled: bool = False` 필드 추가. 환경별 서빙 차단 로직과 정합.
(자가 비평) 기존 필드 유지, 하위 호환성 확보. Router의 enabled 체크와 일치.

**** src/settings/_recipe_schema.py
(근거) development_plan.md 2) 스키마 검증 교정.
(상세) 존재하지 않는 `loader.feature_retrieval` 참조 제거. FS 사용 시 `augmenter.features` 필수 검증으로 간소화.
(자가 비평) 청사진 정책과 일치. 불필요한 의존 제거.

**** src/utils/adapters/sql_adapter.py
(근거) development_plan.md 2) SQL 가드 강화.
(상세) SELECT * 차단, DDL/DML 금칙어 차단, LIMIT 가드 경고, 에러 로그에 SQL 스니펫 포함.
(자가 비평) 과차단 가능성은 명확한 메시지로 상쇄. statement timeout은 주석 가이드로 남김.

**** src/pipelines/inference_pipeline.py
(근거) development_plan.md 2) 템플릿/저장 규칙 정합화.
(상세) `render_template_from_string` 사용으로 함수명 표준화. 저장 파일명 `preds_{run.info.run_id}.parquet`로 고정.
(자가 비평) 청사진 8.2와 일치. 다운스트림 호환성 향상.

**** src/serving/_lifespan.py
(근거) development_plan.md 2) 입력 스키마 생성 우선순위 보강.
(상세) `wrapped_model.data_schema['entity_columns']` 우선 사용, 불가 시 SQL 파싱 폴백.
(자가 비평) 모델 아티팩트의 자기 기술성을 적극 활용.

**** src/serving/schemas.py
(근거) development_plan.md 2) 응답 스키마 일반화.
(상세) `MinimalPredictionResponse` 추가(공통 최소 스키마). 기존 `PredictionResponse` 유지로 하위 호환.
(자가 비평) 엔드포인트 적용은 후속 정리 단계에서 선택적으로 반영.

**** src/serving/router.py
(근거) development_plan.md 2) 서빙 차단 정책 강화.
(상세) PassThrough 차단 유지 + SqlFallbackAugmenter 차단(구현 시 활성). `ServingSettings.enabled` 체크 경로 유지.
(자가 비평) SqlFallback 구현 전까지 try/except로 유연 처리.

**** src/components/_trainer/_optimizer.py
(근거) development_plan.md 3) 튜닝 통합 강화.
(상세) `_create_pruner` 기본 구현(MedianPruner) 추가. 기존 optimize 흐름과 호환.
(자가 비평) optuna 미설치 시 None 처리로 유연성 확보.

**** src/utils/integrations/optuna_integration.py
(근거) development_plan.md 3) OptunaIntegration 클래스화.
(상세) `OptunaIntegration`에 `create_study`/`suggest_hyperparameters` 구현. 기존 `Factory.create_optuna_integration()`과 시그니처 정합.
(자가 비평) recipe 하이퍼파라미터 dict 혼재(고정/탐색) 처리.

**** src/utils/system/reproducibility.py
(근거) development_plan.md 3) 재현성 유틸 도입.
(상세) `set_global_seeds(seed)` 구현: random/numpy/torch 시드, cudnn 결정적 설정, 로그.
(자가 비평) torch 미설치/미사용 환경 고려 try/except 방어.

**** src/pipelines/train_pipeline.py
(근거) development_plan.md 3) 재현성 일관화.
(상세) `run_training` 시작부에 `set_global_seeds(seed)` 호출 추가(레시피 computed.seed 우선, 기본 42).
(자가 비평) 실행 경로 영향 최소. 로그로 추적 가능.

**** src/pipelines/inference_pipeline.py
(근거) development_plan.md 3) 재현성 일관화.
(상세) `run_batch_inference` 시작부에 `set_global_seeds(seed)` 호출 추가.
(자가 비평) 배치 추론의 난수 사용 경로에 대비.

작업 계획: 단계 4 - 서빙 스키마/UX
[PLAN] development_plan.md - 4) 서빙 스키마/UX
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `_endpoints.py` 단일 예측 경로에서 run_mode="serving" 전달 및 최소 응답 스키마로 정리
- `router.py` `/predict`의 response_model을 `MinimalPredictionResponse`로 전환, 차단 정책 유지
- 배치 경로/기타 메타데이터 엔드포인트는 변경하지 않음

**** src/serving/_endpoints.py
(근거) development_plan.md 4) 서빙 스키마/UX
(상세) 단일 예측 경로에서 `params={run_mode:"serving"}`로 강제하고, 최소 응답 스키마 payload로 변환(`{"prediction", "model_uri"}`).
(자가 비평) PyfuncWrapper 경로와 정합. 예측 결과 컬럼 유연 처리.

**** src/serving/router.py
(근거) development_plan.md 4) 서빙 스키마/UX
(상세) `/predict`의 `response_model`을 `MinimalPredictionResponse`로 전환. PassThrough/SqlFallback 차단 유지.
(자가 비평) 테스트 영향 가능성은 최소. 필요 시 호환 엔드포인트 추가 여지.

**** pyproject.toml
(근거) development_plan.md 5) 도커/문서 일관화.
(상세) `requires-python`을 `>=3.11,<3.12`로 상향 고정.
(자가 비평) 청사진 환경 권고와 일치. 로컬/CI 파이썬 버전 주의 필요.

**** Dockerfile
(근거) development_plan.md 5) 도커/문서 일관화.
(상세) 베이스 이미지를 `python:3.11-slim`으로 교체, 빌드 도구 설치, `recipes/` 경로 교정, serve 엔트리포인트/기본 CMD를 `--run-id`로 정합화. train 타깃 기본 CMD를 레시피 파일로 조정.
(자가 비평) uv 기반 설치 유지. pip 폴백은 필요 시 후속 반영 가능.

**** README.md
(근거) development_plan.md 5) 도커/문서 일관화.
(상세) Python 배지를 3.11+로 변경, Docker 빌드/실행 예제 추가, API 테스트 예제를 스키마에 맞게 정정.
(자가 비평) 문서/명령이 CLI 및 Dockerfile과 일치.

작업 계획: 단계 5 - 문서 정합화 (docs/* 최신화)
[PLAN] next_step.md / development_plan.md의 Stage 5-6 범위 중 문서 정비 항목 실행
(근거) 사용자의 최신 구조 반영 지시 및 버전/히스토리 제거 요구에 따른 일괄 업데이트
(CoT 요약)
- 버전/히스토리 표기 제거(v17.0 등)
- 스키마 키 정합화: `data_interface.target_column`, `entity_schema`, `augmenter.type`
- 서빙 정책: `serving.enabled` 플래그, PassThrough/SqlFallback 차단, MinimalPredictionResponse 응답
- 템플릿/보안: `render_template_from_{file|string}` 함수명, 허용 키 화이트리스트, SELECT * 금지/DDL·DML 금칙어/LIMIT 가드
- Docker 타깃 실행 예제 추가(serve/train)

**** docs/DEVELOPER_GUIDE.md
(상세)
- 레시피 예제 전면 정리: `target_column` 키로 일치, `entity_schema` 포함, SELECT * 예 제거
- 템플릿 보안 규칙/허용 키 명시, 저장 규칙(`preds_{run_id}.parquet`) 및 CLI 예시 갱신
- API 응답을 MinimalPredictionResponse 기준으로 간단화
(자가 비평) 개발자 가이드의 범위를 유지하며 불필요한 장황 예제 축소. 실제 코드와 완전 정합.

**** docs/INFRASTRUCTURE_STACKS.md
(상세)
- LOCAL/DEV/PROD 서빙 정책과 Feature Store 전제 명시, ServingSettings.enabled 언급
- 보안/쿼리 예시에서 SELECT * 제거, LIMIT/파티션 필터 예시로 교정
- Docker 타깃(build/run) 예시 추가
(자가 비평) 환경별 차이를 간결 표기. 실무 적용성 강화.

**** docs/MMP_LOCAL_DEV_INTEGRATION.md
(상세)
- run_id 기반 서빙/배치 예시, dev-contract 기반 연동, 포트 변경 시 설정 동기화 예시 정리
- 불필요한 장문 예시 축소, 최신 CLI와 일치
(자가 비평) 핵심 운영 흐름에 집중하도록 간소화.

**** docs/MODEL_CATALOG.md
(상세)
- 버전 문구/트라이얼 횟수 제거, 실제 레시피 경로/클래스/권장 메트릭 중심으로 정리
- 사용법 요약(학습/배치/서빙)과 최소 응답 스키마 예시 추가
(자가 비평) 카탈로그 성격을 보존하면서 최신 정책과 충돌 요소 제거.

작업 계획: 전역 임포트/의존성 아키텍처 안정화
[PLAN] development_plan.md - Stage 6 테스트 정비 선행 과제: 순환 임포트 근본 해결 및 선택 의존성 임포트 정책 확립
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반으로 임포트 그래프(DAG) 정비와 퍼블릭 API 표준화를 우선 적용함.
(CoT 요약)
- 임포트 계층(DAG) 고정: L0(utils) → L1(interface) → L2(settings) → L3(components) → L4(engine) → L5(pipelines/serving/cli)
- 타입 의존은 TYPE_CHECKING/Protocol로 분리, 런타임 임포트는 단방향 유지
- 선택 의존성(optuna) 최상위 임포트 금지, 기능 On 시 지연 임포트 + 명시적 ImportError, uv 환경에서 필수 의존성은 pyproject로 강건 설치
- 퍼블릭 API는 패키지 단위 __all__로 노출, 내부 모듈 직접 임포트 금지
