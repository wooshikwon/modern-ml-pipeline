### 요약(Concise Summary)
- 전역 임포트/의존성 아키텍처 안정화 완료
  - 컴포넌트→엔진 단절(의존 주입), `bootstrap(settings)` 정착, 선택 의존(Optuna) 조건화
  - import-linter 계약 추가 및 위반 0
- 파이프라인/서빙 경로 안정화
  - `Trainer` factory_provider 주입 구조로 전환, `train_pipeline` 반영
  - 로컬 서빙 정책(augmenter pass_through 차단) 테스트 green
  - DEV FS 테스트는 dev 스택 미기동 시 자동 스킵 적용

### 바로 다음 할 일(Next)
1) 테스트 오버홀
- 레거시 테스트(`tests/components/*`, `tests/environments/*`, `tests/integration/*`) 전면 교체
- 공개 API 기준 재작성(Trainer/Preprocessor/Serving)
- 파이프라인(storage+CSV) 및 서빙(로컬 503/DEV 200) 테스트 체계 확립

2) 문서 보강
- DEV 스택 기동/정지/상태/테스트 가이드 강화
- 테스트 마커/스킵 및 import-linter 사용법 추가

3) CI 파이프라인 구성
- lint-imports → 유닛/파이프라인 → 서빙 로컬 정책 → (옵션) DEV 스택 의존 단계 분리

4) 전처리 임시 가드 제거
- FS 경로 안정화 후 `Preprocessor.transform`의 누락 컬럼 0 채움 임시 로직 제거 및 회귀 확인
