# 🚀 Modern ML Pipeline CLI 개발 계획 (Index)

## 📋 Executive Summary

**완전한 마이그레이션 라이프사이클 (6 Phase)**:
- Phase 0-3: 새 시스템 구축 및 통합
- Phase 4: 레거시 코드 Deprecation
- Phase 5: 레거시 코드 완전 제거

**5단계 사용자 플로우 기반 CLI 시스템**:
1. **Init** → 2. **Get-Config** → 3. **System-Check** → 4. **Get-Recipe** → 5. **Train**

**핵심 원칙**: Recipe(논리)와 Config(물리) 완전 분리

---

## 📁 문서 구조

### 개발 Phase 문서

#### 🔨 구축 단계 (Phase 0-3)
- 📄 [**Phase 0: Settings 호환성**](./PHASE_0_SETTINGS_COMPATIBILITY.md) - ✅ 완료
  - Settings 로더 하위 호환성 패치
  - env_name 파라미터 지원
  - 완료일: 2025-08-31

- 📄 [**Phase 1: Get-Config 명령어**](./PHASE_1_GET_CONFIG_COMMAND.md) - ✅ 완료
  - 대화형 환경 설정 생성
  - 동적 .env 템플릿 생성
  - 완료일: 2025-08-31
  - [완료 보고서](./PHASE_1_COMPLETION_REPORT.md)

- 📄 [**Phase 2: --env-name 통합**](./PHASE_2_ENV_NAME_INTEGRATION.md) - ✅ 완료
  - 모든 실행 명령어 수정
  - 환경변수 로더 구현
  - 완료일: 2025-08-31
  - [완료 보고서](./PHASE_2_COMPLETION_REPORT.md)

- 📄 [**Phase 3: 테스트 및 문서화**](./PHASE_3_TESTING_AND_DOCS.md) - ✅ 완료
  - E2E 테스트
  - 사용자 가이드
  - 완료일: 2025-08-31
  - [완료 보고서](./PHASE_3_COMPLETION_REPORT.md)

#### 🔄 마이그레이션 단계 (Phase 4-5)
- 📄 [**Phase 4: Deprecation**](./PHASE_4_DEPRECATION.md) - ✅ 완료
  - 레거시 코드에 경고 추가
  - 마이그레이션 도우미 구현
  - 완료일: 2025-08-31
  - [완료 보고서](./PHASE_4_COMPLETION_REPORT.md)

- 📄 [**Phase 5: Cleanup**](./PHASE_5_CLEANUP.md) - ✅ 완료
  - 레거시 코드 완전 제거
  - v2.0 릴리스 준비
  - 완료일: 2025-08-31
  - [완료 보고서](./PHASE_5_COMPLETION_REPORT.md)

### 참고 문서
- 📄 [**리팩토링 가이드**](./REFACTOR_GUIDE.md)
  - 기존 프로젝트 마이그레이션
  - Settings 모듈 리팩토링
  - 롤백 계획

- 📄 [**상세 개발 계획**](./CLI_REDEVELOPMENT_PLAN.md) (전체 아카이브)
  - 초기 설계 및 모든 세부사항
  - 비판적 검증 결과

---

## 🎯 Quick Start

### 개발자용
```bash
# 1. Phase 0 구현 (필수)
cd src/settings
# PHASE_0_SETTINGS_COMPATIBILITY.md 참조

# 2. Phase 1 구현
cd src/cli/commands
# PHASE_1_GET_CONFIG_COMMAND.md 참조

# 3. Phase 2 구현
cd src/cli/utils
# PHASE_2_ENV_NAME_INTEGRATION.md 참조

# 4. Phase 3 테스트
pytest tests/e2e/
# PHASE_3_TESTING_AND_DOCS.md 참조

# 5. Phase 4 Deprecation
python scripts/add_deprecation_warnings.py
# PHASE_4_DEPRECATION.md 참조

# 6. Phase 5 Cleanup (2주 후)
bash scripts/cleanup_legacy.sh
# PHASE_5_CLEANUP.md 참조
```

### 사용자용 (구현 완료 후)
```bash
# 1. 프로젝트 초기화
mmp init --project-name my-project

# 2. 환경 설정
mmp get-config --env-name dev

# 3. 연결 테스트
mmp system-check --env-name dev

# 4. Recipe 생성
mmp get-recipe

# 5. 학습 실행
mmp train --recipe-file recipes/model.yaml --env-name dev
```

---

## 📊 진행 상황

### Phase별 상태
| Phase | 문서 | 구현 | 테스트 | 상태 |
|-------|------|------|--------|------|
| Phase 0 | ✅ | ✅ | ✅ | **완료** (2025-08-31) |
| Phase 1 | ✅ | ✅ | ✅ | **완료** (2025-08-31) |
| Phase 2 | ✅ | ✅ | ✅ | **완료** (2025-08-31) |
| Phase 3 | ✅ | ✅ | ✅ | **완료** (2025-08-31) |
| Phase 4 | ✅ | ✅ | ✅ | **완료** (2025-08-31) |
| Phase 5 | ✅ | ⏳ | ⏳ | **진행 가능** |

### 호환성 점수: 90/100
- ✅ 완전 호환 (70%): Pipeline, Factory, SqlAdapter, Settings 로더
- ⚠️ 수정 필요 (20%): CLI 명령어, 환경변수 치환
- 🔴 위험 요소 (10%): environment.app_env 제거, 마이그레이션

---

## 🔑 핵심 설계 결정

### 1. Recipe-Config 분리
```yaml
# Recipe (논리적)
adapter: "sql"  # 단순한 이름

# Config (물리적)
connection_uri: "bigquery://..."  # 실제 구현
```

### 2. 환경변수 기반 설정
```yaml
# ${VAR_NAME:default} 패턴
connection_uri: "${DB_CONNECTION_URI:postgresql://localhost/db}"
```

### 3. SqlAdapter의 URI 파싱
- `bigquery://` → BigQuery 엔진
- `postgresql://` → PostgreSQL 엔진
- `mysql://` → MySQL 엔진

---

## ⚡ 우선순위

1. **🔴 긴급**: Phase 0 - Settings 호환성 (다른 모든 작업이 의존)
2. **🟡 중요**: Phase 1-2 - Get-Config 명령어 및 --env-name 통합
3. **🟢 일반**: Phase 3 - 테스트 및 문서화
4. **🔵 계획**: Phase 4 - Deprecation (Phase 3 완료 후)
5. **⚫ 장기**: Phase 5 - Cleanup (v2.0 릴리스 시)

---

## 📞 연락처

- 기술 문의: [Settings 호환성 이슈]
- 사용자 지원: [CLI 사용법 문의]
- 버그 리포트: [GitHub Issues]

---

## 📅 타임라인

### 개발 단계
```
Week 1: Phase 0 (Settings 호환성) ← 현재
Week 1-2: Phase 1 (Get-Config)
Week 2: Phase 2 (--env-name)
Week 2-3: Phase 3 (테스트)
```

### 마이그레이션 단계
```
Week 3: Phase 4 (Deprecation 추가)
Week 3-4: 사용자 마이그레이션 기간
Week 5: Phase 5 (Cleanup)
Week 6: v2.0 릴리스
```

---

*Last Updated: 2025-08-31*
*Version: 5.0 (Phase 0-4 완료)*
*Status: Phase 5 진행 가능*
*Total Phases: 6 (Phase 0-5)*
*Completed: Phase 0-4 ✅*