### 목적
- 이번 사이클을 “테스트 오버홀 + 문서 정합화 + CI 린트 정착”에 집중합니다.
- 이미 완료된 작업은 간결 요약만 남기고, 남은 과제만 실행 단위로 명확화합니다.

### 완료 요약
- 전역 임포트/의존성 안정화: 컴포넌트→엔진 단절, `bootstrap(settings)` 정착, 선택 의존성(Optuna) 조건화, import-linter 계약 준수.
- 서빙 정책 정비: 로컬 정책(augmenter pass_through 차단 503) 테스트 green, DEV FS 테스트는 dev 스택 없으면 자동 스킵.
- 파이프라인 정비: `Trainer`의 `factory_provider` 주입 구조 확립, `train_pipeline` 반영.

### 남은 과제(실행 항목)
1) 테스트 오버홀
- 레거시 테스트(`tests/components/*`, `tests/environments/*`, `tests/integration/*`) 전면 교체
- 공개 API 기준으로 재작성(`src.components._trainer.Trainer` with factory_provider, `src.components._preprocessor.Preprocessor`, `src.serving.router`)
- 파이프라인 테스트: storage+CSV 기반의 안정 경로로 `train`/`inference` 검증
- 서빙 테스트 분리 유지: 로컬(정책 503), DEV FS(200, dev 스택 시만)

2) 문서 보강
- DEV 스택 기동/정지/상태/테스트: `./setup-dev-environment.sh start|stop|status|test`
- 테스트 실행 매트릭스와 마커/스킵 가이드(`@requires_dev_stack`, `ensure_dev_stack_or_skip`)
- import-linter 실행 가이드(`uv run lint-imports`)

3) CI 파이프라인(스크립트/워크플로)
- 단계 분리: import-linter → 유닛/파이프라인 → 서빙 정책(로컬) → DEV 스택 의존 테스트(옵션)

4) 전처리 임시 가드 제거
- `Preprocessor.transform`의 “누락 컬럼 0으로 채움” 임시 로직 제거
- FS 경로 안정화 후 회귀 테스트로 검증

### 수용 기준(AC)
- 린트: `uv run lint-imports` 계약 위반 없음
- 서빙: 로컬 정책 테스트 green, DEV FS 테스트는 dev 스택 기동 시 200/미기동 시 skip
- 파이프라인: storage+CSV 기준 `train`/`inference` 테스트 green
- 전처리: 임시 가드 제거 후 회귀 테스트 green

### 실행 순서(권장)
1) 테스트 오버홀 재작성 → 2) 문서 보강 → 3) CI 파이프라인 구성 → 4) 전처리 임시 가드 제거