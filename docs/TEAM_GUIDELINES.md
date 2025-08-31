# 🤝 팀 가이드라인 & 체크리스트 - Modern ML Pipeline

**Phase 4-4.5 성과 보호 및 팀 협업 최적화를 위한 종합 가이드라인**

---

## 📋 목차

1. [코드 리뷰 체크리스트](#-코드-리뷰-체크리스트)
2. [Pull Request 가이드라인](#-pull-request-가이드라인)
3. [릴리즈 체크리스트](#-릴리즈-체크리스트)
4. [개발 워크플로](#-개발-워크플로)
5. [품질 관리 체크리스트](#-품질-관리-체크리스트-phase-4-5-성과-보호)
6. [온보딩 체크리스트](#-온보딩-체크리스트)
7. [긴급 상황 대응](#-긴급-상황-대응)

---

## 📝 코드 리뷰 체크리스트

**모든 PR은 최소 2명의 승인 필요 - Phase 4-5 품질 기준 유지**

### ✅ 기본 체크리스트

**코드 품질 (필수)**
- [ ] **TDD 원칙 준수**: 테스트가 먼저 작성되었는가?
- [ ] **Factory 패턴 사용**: 새로운 테스트에서 Factory 패턴을 사용했는가?
- [ ] **타입 힌트**: 모든 public 함수/클래스에 타입 힌트가 있는가?
- [ ] **Google Style Docstring**: public API에 문서화가 되어있는가?
- [ ] **네이밍 규칙**: `snake_case`, `PascalCase`, `UPPER_CASE` 준수했는가?

**테스트 품질 (필수 - Phase 4-5 기준)**
- [ ] **테스트 마커**: 적절한 pytest 마커가 적용되었는가? (`@pytest.mark.unit`, `@pytest.mark.core`)
- [ ] **테스트 명명**: `test_<컴포넌트>_should_<행동>_when_<조건>` 규칙을 따르는가?
- [ ] **Given/When/Then**: 테스트 구조가 명확한가?
- [ ] **테스트 독립성**: 각 테스트가 독립적으로 실행되는가?

**성능 영향 (Phase 4-5 성과 보호)**
- [ ] **핵심 테스트 성능**: 새로운 `@pytest.mark.core` 테스트의 실행 시간이 적절한가?
- [ ] **Mock Registry 사용**: 적절히 MockComponentRegistry를 활용했는가?
- [ ] **Session-scoped Fixtures**: 불변 데이터에 대해 session 스코프를 사용했는가?

### ✅ 고급 체크리스트

**설계 품질**
- [ ] **Blueprint 원칙**: 설정과 논리의 분리 원칙을 지켰는가?
- [ ] **의존성 주입**: Factory Provider 패턴을 올바르게 사용했는가?
- [ ] **에러 처리**: 적절한 예외 처리와 에러 메시지가 있는가?
- [ ] **보안**: 비밀 정보가 하드코딩되지 않았는가?

**문서화 및 사용성**
- [ ] **사용 예제**: 복잡한 기능에 대한 사용 예제가 있는가?
- [ ] **Breaking Changes**: 호환성을 깨는 변경사항이 문서화되었는가?
- [ ] **마이그레이션 가이드**: 기존 사용자를 위한 마이그레이션 정보가 있는가?

---

## 🔄 Pull Request 가이드라인

### PR 생성 전 체크리스트

```bash
# 1. 로컬 테스트 통과 확인
uv run pytest -m "core and unit" -v  # 핵심 테스트 (3초 이내)
uv run pytest tests/unit/ -q         # 전체 단위 테스트

# 2. 코드 품질 확인
uv run pre-commit run --all-files

# 3. 타입 체크
uv run mypy src

# 4. Phase 4-5 성과 검증
./scripts/verify_test_coverage.sh
```

### PR 제목 및 설명 작성법

**제목 형식**: `<type>(<scope>): <description> (Task-ID)`

```
feat(components): add data preprocessing pipeline (P05-5)
fix(trainer): resolve memory leak in batch processing (P05-6)
docs(readme): update installation guide for new features (P05-7)
test(factories): add regression data factory tests (P05-8)
refactor(engine): optimize model loading performance (P05-9)
```

**PR 템플릿**:

```markdown
## 🎯 목적
- 무엇을 왜 변경했는지 간단히 설명

## 🔄 변경 사항
- [ ] 새로운 기능 추가
- [ ] 버그 수정
- [ ] 성능 개선
- [ ] 리팩토링
- [ ] 문서 업데이트
- [ ] 테스트 추가/개선

## 🧪 테스트 전략
- [ ] 새로운 단위 테스트 추가 (Factory 패턴 사용)
- [ ] 기존 테스트 수정
- [ ] 통합 테스트 필요
- [ ] 성능 테스트 필요

## 📊 Phase 4-5 성과 유지 확인
- [ ] 핵심 테스트 실행 시간 3초 이내 유지
- [ ] Factory 패턴 일관성 유지
- [ ] Mock Registry LRU 캐싱 활용
- [ ] 전체 단위 테스트 통과율 100% 유지

## ✅ 체크리스트
- [ ] 로컬에서 모든 테스트 통과
- [ ] 코드 리뷰 가이드라인 준수
- [ ] 문서화 완료 (필요시)
- [ ] Breaking changes 문서화 (해당시)
- [ ] 성능 영향 검토 완료
```

### 머지 기준

**필수 조건**:
1. **자동 체크 통과**: CI/CD 파이프라인 모든 단계 통과
2. **코드 리뷰 승인**: 최소 2명의 리뷰어 승인
3. **테스트 커버리지**: 새로운 코드에 대한 테스트 존재
4. **성능 회귀 없음**: Phase 4-5 성능 기준 유지

**머지 방식**:
- **Squash and Merge** (기본): 피처 브랜치의 여러 커밋을 하나로 통합
- **Merge Commit**: 중요한 피처나 릴리즈 브랜치
- **Rebase and Merge**: 히스토리가 중요한 핫픽스

---

## 🚀 릴리즈 체크리스트

### Pre-Release 체크리스트

**코드 품질 검증**
- [ ] **전체 테스트 스위트 통과**: `uv run pytest tests/`
- [ ] **성능 벤치마크 확인**: `./scripts/verify_test_coverage.sh`
- [ ] **메타 테스트 통과**: `uv run pytest tests/meta/ -v`
- [ ] **코드 커버리지**: 최소 30% 이상 (핵심 모듈 집중)

**문서 업데이트**
- [ ] **CHANGELOG.md** 업데이트
- [ ] **README.md** 새로운 기능 반영
- [ ] **API 문서** 업데이트
- [ ] **마이그레이션 가이드** 작성 (Breaking Changes 시)

**환경 테스트**
- [ ] **로컬 환경**: `APP_ENV=local` 테스트
- [ ] **개발 환경**: `APP_ENV=dev` 연동 테스트
- [ ] **Docker 이미지**: 빌드 및 실행 테스트
- [ ] **종속성 검증**: `uv sync --locked` 성공

### Release 체크리스트

**버전 관리**
- [ ] **버전 태그 생성**: `git tag v1.x.y`
- [ ] **릴리즈 노트** 작성 (GitHub Releases)
- [ ] **Docker 이미지** 태깅 및 푸시
- [ ] **PyPI 패키지** 업로드 (해당시)

**배포 후 확인**
- [ ] **핵심 기능 스모크 테스트**
- [ ] **성능 모니터링** 3시간 관찰
- [ ] **에러 로그** 모니터링
- [ ] **롤백 계획** 준비 완료

---

## ⚙️ 개발 워크플로

### 브랜치 전략

```
main (production-ready)
├── develop (integration branch)
├── feature/add-new-processor (P05-10)
├── feature/improve-performance (P05-11)
├── hotfix/critical-bug-fix (P05-12)
└── release/v1.2.0 (P05-13)
```

**브랜치 명명 규칙**:
- `feature/brief-description` - 새로운 기능
- `bugfix/issue-description` - 버그 수정
- `hotfix/critical-issue` - 긴급 수정
- `release/v1.x.y` - 릴리즈 준비
- `docs/section-update` - 문서 업데이트

### 개발 사이클

**1. 기능 개발 시작**
```bash
# develop 브랜치에서 시작
git checkout develop
git pull origin develop

# 새 피처 브랜치 생성
git checkout -b feature/add-data-validator
```

**2. TDD 개발**
```bash
# RED: 실패하는 테스트 작성
uv run pytest tests/unit/components/test_data_validator.py::TestDataValidator::test_validate -v
# FAILED (예상)

# GREEN: 최소 구현
# src/components/data_validator.py 구현

# REFACTOR: 코드 개선
# 리팩토링 후 테스트 재실행
```

**3. 지속적 검증**
```bash
# 개발 중 빠른 검증 (3초)
uv run pytest -m "core and unit" -v

# 커밋 전 전체 검증
uv run pre-commit run --all-files
uv run pytest tests/unit/ -q
```

**4. PR 및 머지**
```bash
# PR 생성 전 최종 확인
./scripts/verify_test_coverage.sh

# PR 생성 및 리뷰
git push origin feature/add-data-validator
```

---

## 🛡️ 품질 관리 체크리스트 (Phase 4-5 성과 보호)

### 매일 품질 체크

**개발자 개인**
- [ ] **빠른 테스트**: `uv run pytest -m "core and unit" -v` (3초 이내)
- [ ] **Factory 패턴**: 새로운 테스트에서 TestDataFactory/SettingsFactory 사용
- [ ] **타입 체크**: `uv run mypy src` 통과

**팀 공동**
- [ ] **CI 상태**: GitHub Actions 모든 체크 통과
- [ ] **테스트 안정성**: 단위 테스트 100% 통과율 유지
- [ ] **성능 모니터링**: 핵심 테스트 실행 시간 추적

### 주간 품질 리뷰

**매주 금요일 체크**
- [ ] **종합 성과 검증**: `./scripts/verify_test_coverage.sh` 실행
- [ ] **메타 테스트**: `uv run pytest tests/meta/ -v` 통과
- [ ] **코드 커버리지**: 트렌드 모니터링
- [ ] **기술부채**: 리팩토링 필요 영역 식별

**Phase 4-5 성과 지표 모니터링**
- [ ] **77% 성능 향상 유지**: 핵심 테스트 3초 이내
- [ ] **100% 단위 테스트 안정화**: 79/79 테스트 통과
- [ ] **Factory 패턴 적용률**: 90% 이상 유지
- [ ] **Mock Registry 효율성**: 캐시 히트율 70% 이상

### 월간 품질 감사

**매월 첫째 주**
- [ ] **전체 테스트 실행**: 느린 테스트 및 통합 테스트 포함
- [ ] **성능 벤치마크**: 기준선 대비 성능 변화 분석
- [ ] **코드 품질 메트릭**: 복잡도, 중복도, 유지보수성 평가
- [ ] **보안 검토**: 의존성 취약성 및 보안 패치 확인

---

## 👥 온보딩 체크리스트

### 새 팀원 첫째 날

**환경 설정 (2시간)**
- [ ] **저장소 클론**: `git clone` 및 `uv sync`
- [ ] **IDE 설정**: VS Code + 권장 확장프로그램
- [ ] **환경변수**: `.env` 파일 설정
- [ ] **테스트 실행**: `uv run pytest -m "core and unit" -v` 성공

**팀 규칙 숙지 (1시간)**
- [ ] **[DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)** 읽기
- [ ] **[CLAUDE.md](../CLAUDE.md)** TDD 원칙 이해
- [ ] **이 문서** 팀 가이드라인 숙지
- [ ] **Slack/Discord** 채널 가입

### 첫째 주

**학습 과제**
- [ ] **Factory 패턴 이해**: TestDataFactory, SettingsFactory 사용법 숙지
- [ ] **TDD 실습**: 간단한 기능 RED-GREEN-REFACTOR로 구현
- [ ] **코드 리뷰 참여**: 2-3개 PR 리뷰 경험
- [ ] **첫 번째 기여**: Good First Issue 해결

**멘토링 체크포인트**
- [ ] **Day 3**: 개발 환경 및 기본 워크플로 점검
- [ ] **Day 5**: 첫 번째 PR 리뷰 (멘토와 페어)
- [ ] **Day 7**: 주간 회고 및 질문 정리

### 첫째 달

**역량 개발**
- [ ] **Blueprint 원칙**: 설계 철학 깊이 이해
- [ ] **성능 최적화**: Phase 4-5 최적화 기법 학습
- [ ] **복잡한 기능**: 멀티 컴포넌트 기능 개발
- [ ] **릴리즈 경험**: 릴리즈 프로세스 참여

**자립 지표**
- [ ] **독립적 개발**: 멘토링 없이 기능 개발 가능
- [ ] **코드 리뷰**: 다른 팀원 PR 리뷰 가능
- [ ] **문제 해결**: 일반적인 이슈 독립적 해결
- [ ] **지식 공유**: 팀 미팅에서 기술 내용 발표

---

## 🚨 긴급 상황 대응

### 프로덕션 이슈 대응

**심각도 1 (서비스 중단)**
1. **즉시 대응** (5분 이내)
   - Slack `#emergency` 채널 알림
   - 온콜 개발자 호출
   - 사고 관리자 지정

2. **초기 조치** (15분 이내)
   - 로그 및 모니터링 확인
   - 롤백 여부 결정
   - 임시 해결책 적용

3. **근본 해결** (1시간 이내)
   - 핫픽스 브랜치 생성: `hotfix/critical-production-fix`
   - 최소한의 테스트로 빠른 검증
   - 긴급 배포 및 모니터링

**심각도 2 (기능 장애)**
- 정상 개발 프로세스 단축 버전
- 빠른 리뷰 (1명 승인)
- 집중 테스트 및 모니터링

### 테스트 실패 대응

**Phase 4-5 성과 보호**
```bash
# 핵심 테스트 실패 시
uv run pytest -m "core and unit" -v --tb=short
# 원인 분석 후 즉시 수정

# 전체 테스트 실패 시  
uv run pytest tests/unit/ -x -v  # 첫 실패에서 중단
# Factory 패턴 관련 이슈인지 확인

# 성능 회귀 감지 시
./scripts/verify_test_coverage.sh
# 3초 기준 초과시 즉시 최적화
```

**에스컬레이션 기준**
- 핵심 테스트 10초 초과 → 팀 리드 에스컬레이션
- 단위 테스트 통과율 95% 미만 → 개발 중단, 원인 분석
- 메타 테스트 실패 → 테스트 품질 시스템 점검

---

## 📊 성과 지표 추적

### 개발 생산성 지표

**일일 추적**
- **핵심 테스트 실행 시간**: 목표 3초 이내
- **단위 테스트 통과율**: 목표 100%
- **PR 머지까지 시간**: 목표 24시간 이내
- **코드 리뷰 피드백 수**: 품질 개선 지표

**주간 추적**  
- **Factory 패턴 사용률**: 목표 90% 이상
- **Mock Registry 캐시 히트율**: 목표 70% 이상
- **코드 커버리지**: 트렌드 모니터링
- **기술부채 해결**: 주간 목표 설정

### 팀 협업 지표

**협업 효율성**
- **코드 리뷰 소요 시간**: 평균 2시간 이내
- **PR 블로킹 시간**: 24시간 이상 0건
- **리뷰 피드백 품질**: 건설적 피드백 비율
- **온보딩 성공률**: 신규 팀원 1주일 내 자립률

---

**🎯 이 가이드라인은 Phase 4-4.5 성과(77% 성능 향상, 100% 테스트 안정화)를 보호하고 팀의 생산성을 극대화하기 위해 설계되었습니다.**

**📅 정기 업데이트**: 매 분기 팀 회고를 통해 가이드라인을 개선합니다.