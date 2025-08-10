### 목적
- 본 문서는 “테스트 오버홀 + 문서 정합화 + CI 린트 정착”을 위한 실행 가이드입니다. 완료된 사항은 제외하고, 남은 항목만 구현 단위로 기술합니다.

---

## A) 테스트 오버홀

- 작업
  - 레거시 테스트 전면 교체(`tests/components/*`, `tests/environments/*`, `tests/integration/*`)
  - 공개 API 기준으로 재작성
    - Trainer: `src.components._trainer.Trainer(settings, factory_provider=...)`
    - Preprocessor: `src.components._preprocessor.Preprocessor`
    - Serving: `src.serving.router`(정적 `/predict`), 정책 503 체크
  - 파이프라인 테스트
    - storage+CSV 기반으로 `train`/`batch-inference` 정상 경로 검증
  - 서빙 테스트 분리
    - 로컬: 정책 503 green
    - DEV: Feature Store 경로 200, `ensure_dev_stack_or_skip`로 미기동 시 skip

- 수용 기준(AC)
  - `tests/pipelines/*`, `tests/serving/*` green
  - DEV FS 테스트는 dev 스택 기동 시 200/미기동 시 자동 skip

---

## B) 문서 보강

- 작업
  - DEV 스택 운영 가이드 보강
    - `./setup-dev-environment.sh start|stop|status|test`
  - 테스트 실행 매트릭스/마커 가이드
    - `@pytest.mark.requires_dev_stack`, `ensure_dev_stack_or_skip`
  - 린트/품질 가이드
    - `uv run lint-imports`

- 수용 기준(AC)
  - README/DEVELOPER_GUIDE/MMP_LOCAL_DEV_INTEGRATION 최신 구조 반영 및 명령 검증

---

## C) CI 파이프라인(워크플로/스크립트)

- 작업
  - 단계 분리 실행
    - Step1: import-linter
    - Step2: 유닛/파이프라인 테스트
    - Step3: 서빙 로컬 정책 테스트
    - Step4: DEV 스택 의존 테스트(옵션/매뉴얼 트리거)

- 수용 기준(AC)
  - 린트 위반 시 실패, 테스트 단계별 결과 가시화

---

## D) 전처리 임시 가드 제거

- 작업
  - `src/components/_preprocessor/_preprocessor.py::transform`의 “누락 컬럼 0 채움” 임시 로직 제거
  - Feature Store 경로의 입력 스키마 안정화 후 제거 시점 확정

- 수용 기준(AC)
  - 제거 후 회귀 테스트 green(로컬/DEV 스택 모두)

---

## 실행 순서
1) 테스트 오버홀 재작성
2) 문서 보강 반영
3) CI 파이프라인 적용
4) 전처리 임시 가드 제거 및 회귀 검증


