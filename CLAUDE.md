# Claude Guide – Vibe Coding 프로젝트 지침

Claude는 이 문서를 세션 시작 시 읽습니다.  
**목표: 짧은 요약만 로드하고, 세부 지침은 `.claude/` 문서로 분리**합니다.

---

## 1) 프로젝트 스냅샷
- Python 기반 AI/MLOps & AI Agent
- 스타일: **PEP8**, 네이밍: snake_case / PascalCase / UPPER_CASE
- 문서화: **Google Style Docstring**, 타입 힌트 **가능한 모두**
- 테스트: **pytest**, 프로세스: **TDD**
- 패키지/실행: **uv** (`uv sync`, `uv run ...`)
- 청사진: [`BLUEPRINT.md`](./.claude/BLUEPRINT.md)

---

## 2) 세션 부팅 체크리스트 (Claude가 먼저 수행)
1. **작업 계획 TL;DR 요약** → 오늘 처리할 Task 선택  
2. **테스트 먼저(RED)** → 최소 구현(GREEN) → 리팩터  
3. 커밋/PR에 **Task ID** 포함 (예: `P02-1`)
> 상세 플랜은 `.claude/DEV_PLANS.md` 참고

---

## 3) 작업 계획 (Working Plan)
@.claude/DEV_PLANS_TLDR.md
- Full: [.claude/DEV_PLANS.md](./.claude/DEV_PLANS.md)

---

## 4) 스타일 / 테스트 / 환경 — TL;DR만 자동 로드
**Style TL;DR**  
@.claude/STYLEGUIDE_TLDR.md  
Full: [.claude/STYLEGUIDE.md](./.claude/STYLEGUIDE.md)

**Testing TL;DR**  
@.claude/TESTING_TLDR.md  
Full: [.claude/TESTING.md](./.claude/TESTING.md)

**Environment TL;DR**  
@.claude/ENVIRONMENT_TLDR.md  
Full: [.claude/ENVIRONMENT.md](./.claude/ENVIRONMENT.md)

---

## 5) 골든 커맨드 (자주 쓰는 명령)
- 초기 동기화: `uv sync`
- 테스트: `uv run pytest -q` / 커버리지: `uv run pytest --cov=app --cov-report=term-missing -q`
- 패키지 추가/제거: `uv add <pkg>` / `uv remove <pkg>`
- 앱/스크립트 실행: `uv run python <file.py> [args]`

---

## 6) Git & 워크플로우
- 브랜치: `feature/<설명>`에서 작업, `main` 보호
- 커밋: Conventional Commits (예: `feat: add user login (P02-1)`)
- 작은 단위 커밋 → PR 생성 → 자동 테스트 통과 후 머지

---

## 7) MUST 규칙 (위반 시 중단)
- **테스트 없는 구현 금지**(TDD 우선), 커버리지 **≥ 90%**
- **타입 힌트 + Google Docstring** 필수(공개 함수/클래스/모듈)
- **uv만 사용**(글로벌 pip 금지), 비밀은 **.env** 사용
- 스타일/테스트 실패 PR은 **머지 불가**

---
