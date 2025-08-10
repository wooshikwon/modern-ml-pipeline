### 목적
- 이번 사이클의 남은 범위를 “전역 임포트/의존성 안정화 + 테스트 전면 정비(Stage 6)”로 일원화합니다.
- 이미 완료된 항목은 [완료]로 고정 표시하고, 미완료는 실행 단위로 쪼개어 [진행]/[예정]으로 명확화합니다.

### 완료된 항목
- [완료] Stage 1: 치명적 런타임 버그 및 인터페이스 정합성 제거
- [완료] Stage 2: 정책/보안/청사진 정합화(템플릿/SQL/서빙)
- [완료] Stage 3: 재현성/튜닝(전역 시드, OptunaIntegration, Optimizer)
- [완료] Stage 4: 서빙 스키마/UX(서빙 차단 정책, MinimalPredictionResponse)
- [완료] Stage 5: 도커/문서 일관화(Python 3.11, Dockerfile/README)
- [완료] Stage 6-1: 부트스트랩 도입(`engine.bootstrap(settings)`) 및 파이프라인 엔트리 적용
- [완료] Stage 6-2: 레시피-주도 어댑터 선택(파이프라인에서 어댑터 인자 제거)
- [완료] Stage 6-3: `StorageAdapter` CSV 지원 추가
- [완료] Stage 6-4: `Trainer`의 데이터 준비 경로 정리(DataHandler 직접 의존 제거)

### 진행 항목 — Stage 6: 전역 안정화 + 테스트 개편
- [진행] Stage 6-5: 의존성 검증 유틸 `validate_dependencies(settings)` 도입 및 `bootstrap` 연동(미설치 시 ImportError로 즉시 실패)
- [진행] Stage 6-6: `Factory` 내부 레지스트리 지연등록(fallback) 제거 → 부트스트랩 누락 시 명확 실패
- [진행] Stage 6-7: 데이터 분할 stratify 가드(소표본/불균형 시 stratify 비활성)
- [예정] Stage 6-8: 서빙 초기화 경로에 `bootstrap(settings)` 보장(예: `setup_api_context`)
- [예정] Stage 6-9: Import-linter 계약 추가(계층 경계 규율)
- [진행] Stage 6-10: 테스트 픽스처 확장(>=50행, 타깃 클래스 최소 10+), 파이프라인 테스트는 storage 기반으로 고정
- [예정] Stage 6-11: SQL/Feast는 `@requires_dev_stack` 통합 테스트로만 실행(유닛은 경로/가드만 검증)
- [진행] Stage 6-12: 불필요한 예외/우회 로직 제거(Factory fallback, 임시 sqlite 연결 등)

### 핵심 정합화 포인트
- 임포트/경로: 최신 공개 API 기준(`src.serving.router`, `src.engine.factory`, `src/utils/adapters/*`)으로 교정
- 레시피-주도: 데이터 어댑터·로더 선택은 레시피가 단일 진실원천
- 의존성: 기능 활성에 따른 패키지 필수 검사(미설치 시 즉시 실패; 우회 금지)
- 테스트: 유닛/파이프라인은 storage + CSV/Parquet 기반, SQL/Feast는 통합 레벨로 분리
- 보안/정책: 템플릿 화이트리스트, SELECT * 차단, DDL/DML 금칙어, 서빙 차단 정책 유지

### 수용 기준(AC)
- LOCAL: `tests/pipelines/*` 및 `tests/utils/*` green, 작은 데이터에서도 분할 실패 없음
- 의존성 검증: 활성 기능에 필요한 패키지 미설치 시 ImportError로 즉시 실패(패키지명 포함)
- 부트스트랩 규율: `Factory` 내부 fallback 제거로, 부트스트랩 누락 시 명확 실패
- Import-linter: 계약 위반 없음(추가 후 CI 기준)
- 픽스처 일관성: storage 기반 데이터로 파이프라인 테스트 안정 통과

### 우선순위/순서
1) 의존성 검증 유틸 추가 및 `bootstrap` 연동(6-5)
2) Factory fallback 제거(6-6)
3) 분할 stratify 가드(6-7)
4) 테스트 픽스처 보강 및 파이프라인 테스트 안정화(6-10)
5) 서빙 부트스트랩 보장(6-8)
6) Import-linter 계약 추가(6-9)
7) SQL/Feast 통합 테스트 경계 정리(6-11)
8) 불필요 예외/우회 제거 마무리(6-12)

### 산출물
- 코드: `src/utils/system/dependencies.py`, `src/engine/factory.py`, `src/components/_trainer/_data_handler.py`, 서빙 초기화 경로, `pyproject.toml`(import-linter)
- 테스트/픽스처: `tests/pipelines/*`, `tests/utils/*`, `tests/fixtures/data/*`