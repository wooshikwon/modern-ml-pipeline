# Claude Guide – Vibe Coding 프로젝트 지침

---

## 1) 프로젝트 스냅샷

* Python 기반 **AI/MLOps & AI Agent**
* 스타일: **PEP8**, 네이밍: `snake_case` / `PascalCase` / `UPPER_CASE`
* 문서화: **Google Style Docstring**, **타입 힌트 최대화**
* 테스트: **pytest**, 프로세스: **TDD**
* 패키지/실행: **uv** (`uv sync`, `uv run ...`)
* 청사진: [`BLUEPRINT.md`](./.claude/BLUEPRINT.md)
* 관련 가이드(단일 compact):

  * 테스트: [`TESTING.md`](./.claude/TESTING.md)
  * 스타일: [`STYLEGUIDE.md`](./.claude/STYLEGUIDE.md)
  * 환경: [`ENVIRONMENT.md`](./.claude/ENVIRONMENT.md)

---

## 2) 세션 부팅 체크리스트 (Claude가 먼저 수행)

> **원칙**: **RED(실패 테스트)** → **GREEN(최소 구현)** → **REFACTOR**. 테스트 없는 변경 금지.

1. **동기화**: `uv sync`
2. **정적검사/포맷 확인**
   `uv run ruff check . && uv run black --check . && uv run isort --check-only .`
3. **타입체크**: `uv run mypy src`
4. **기본 테스트 스위트**: `uv run pytest -q -m "not slow and not integration"`
5. 실패 시 행동: **실패 테스트 보강/추가 → 최소 구현으로 통과 → 리팩터 → 전체 재실행**
6. 커밋/PR에 **Task ID** 포함 (예: `feat(auth): add login (P02-1)`)
   세부 플랜은 `.claude/DEV_PLANS.md` 참고

---

## 3) 작업 계획 (Working Plan)

* 요약/우선순위: `.claude/DEV_PLANS.md`의 상단 실행 블록
* 근거/세부는 같은 파일의 `<details>` 영역

---

## 4) 골든 커맨드

### 기본 도구
* 동기화: `uv sync`
* 정적검사/포맷: `uv run ruff check .` / `uv run black --check .` / `uv run isort --check-only .`
* 타입체크: `uv run mypy src`
* pre-commit 전체 실행: `uv run pre-commit run -a`
* 패키지 추가/제거: `uv add <pkg>` / `uv remove <pkg>`

### 테스트 실행 전략 (Phase 4 최적화)
* **빠른 개발용** (핵심만): `uv run pytest -m "core and unit" -v`
* **표준 CI** (기본): `uv run pytest -q -m "not slow and not integration"`
* **성능 최적화** (병렬): `uv run pytest -n auto tests/unit/ -v`  
* **커버리지 검증**: `uv run pytest --cov=src --cov-report=term-missing --fail-under=90 -q`
* **전체 단위**: `uv run pytest tests/unit/ -v`
* **핵심+통합**: `uv run pytest -m "(core and unit) or integration" -v`

---

## 5) Git & 워크플로

* 브랜치: `feature/<설명>`에서 작업, `main` 보호
* 커밋: Conventional Commits + **Task ID 필수** (예: `feat(auth): add login (P02-1)`)
* 작은 단위 커밋 → PR 생성 → **자동 테스트 통과 + 리뷰 2인 승인** 후 머지

---

## 6) MUST 규칙 (위반 시 중단)

* **테스트 없는 구현 금지**, 커버리지 **≥ 90%**
* **타입 힌트 + Google Docstring** (공개 함수/클래스/모듈)
* **uv만 사용**(글로벌 pip 금지), 비밀은 **.env**/**Secret Manager**로 주입
* ruff/black/isort/mypy/pytest/커버리지 **실패 PR은 머지 불가**

---

## 7) TDD 테스트 운영 규칙

* **마커**: `@pytest.mark.slow`, `@pytest.mark.integration`  → 기본 스위트에서 제외
* **명명**: `test_<모듈>__<행동>__<기대>()` (Given/When/Then 주석 권장)
* **픽스처**: 고정 데이터 `tests/fixtures/`; 외부 I/O는 `monkeypatch`/stub으로 격리
* **속성기반**(선택): 핵심 순수함수에 `hypothesis` 적용
* **피라미드**: 단위 > 통합 > E2E. 외부 네트워크/파일/IaC 의존은 가짜/임시 디렉토리 사용

---

## 8) CI 게이트 (머지 조건)

* ruff/black/isort **통과**
* mypy **통과**
* pytest 기본 스위트 **통과**: `-m "not slow and not integration"`
* 커버리지 **Fail-under 90** 강제
* PR 템플릿 체크리스트 **모두 충족**
* Nightly 워크플로에서 `slow/integration` 전체 실행 (스케줄/수동)

---

## 9) 환경/비밀

* 로컬: `.env` 사용, **커밋 금지** → `.env.example` 유지(필수 키 주석 포함)
* CI/운영: GitHub Actions **Secrets** 또는 클라우드 **Secret Manager**로 주입
* 재현성: `PYTHONHASHSEED=0` 및 랜덤 시드 고정 원칙 명시

---

## 10) 리포 구조 합의

```
repo/
├── src/<pkg>/
│   ├── __init__.py   # 공개 API 노출
│   └── ...
├── tests/
│   ├── unit/         # 기본 스위트 대상 (-m "not slow and not integration")
│   ├── integration/  # @pytest.mark.integration
│   └── e2e/
├── .claude/
│   ├── BLUEPRINT.md
│   ├── DEV_PLANS.md
│   ├── TESTING.md
│   ├── STYLEGUIDE.md
│   ├── ENVIRONMENT.md
│   └── CI.md (선택: CI 상세 배경)
└── .github/
    └── workflows/
```

---

## 11) Claude가 각 문서를 읽는 방법 (단일 compact 기준)

* 각 문서 상단에 다음 **실행 블록**을 위치:

```md
<!-- CLAUDE:BEGIN -->
(여기까지가 실행 블록: MUST/게이트/골든 커맨드/요약 규칙)
<!-- CLAUDE:END -->
```

* 상세 내용은 아래처럼 접어 보관:

```md
<details>
  <summary>상세(이유/예시/트러블슈팅)</summary>
  ...
</details>
```

---

## 12) 부록 — 바로 쓸 스니펫

### 12.1 PR 템플릿 (`.github/pull_request_template.md`)

```md
## 목적
- 무엇을, 왜 변경했는지 TL;DR

## 변경 사항
- 핵심 변경 요약(불릿)

## 테스트 전략
- [ ] 단위 테스트 추가/수정 목록
- [ ] 통합/슬로우 테스트 필요 시 방법

## 체크리스트
- [ ] ruff/black/isort 통과
- [ ] mypy 통과
- [ ] pytest 기본 스위트 통과 (`-m "not slow and not integration"`)
- [ ] 커버리지 ≥ 90% (`--fail-under=90`)
- [ ] 공개 API에 Docstring + 사용 예제 1개 이상
- [ ] 커밋 메시지: `type(scope): ... (TaskID)`

## 롤백 플랜
- 문제 발생 시 되돌리는 방법
```

### 12.2 pre-commit 설정 (`.pre-commit-config.yaml`)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.1
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - types-PyYAML
```

### 12.3 GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      - name: Sync
        run: uv sync
      - name: Lint & Format
        run: |
          uv run ruff check .
          uv run black --check .
          uv run isort --check-only .
      - name: Typecheck
        run: uv run mypy src
      - name: Test (default suite)
        run: uv run pytest -q -m "not slow and not integration"
      - name: Coverage Gate
        run: uv run pytest --cov=src --cov-report=term-missing --fail-under=90 -q
```

### 12.4 Nightly 워크플로 (`.github/workflows/nightly.yml`)

```yaml
name: Nightly Tests
on:
  schedule:
    - cron: '0 18 * * *'  # 매일 03:00 KST
  workflow_dispatch:

jobs:
  nightly:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      - name: Sync
        run: uv sync
      - name: Test (slow/integration)
        run: uv run pytest -q -m "slow or integration"
```

### 12.5 도구 설정 예시 (`pyproject.toml` 발췌)

```toml
[tool.ruff]
line-length = 100
select = ["E","F","I","UP","B","SIM","PL"]
ignore = ["E203","E501"]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.11"
warn_unused_ignores = true
warn_return_any = true
disallow_untyped_defs = true
no_implicit_optional = true
```

---

## 13) 메모

* 본 문서는 **단일 compact 정책**을 반영합니다. TL;DR 분리 문서가 존재하지 않음을 전제로 유지/운용하세요. 각 하위 가이드의 상단 실행 블록 범위는 마커로 고정합니다.
