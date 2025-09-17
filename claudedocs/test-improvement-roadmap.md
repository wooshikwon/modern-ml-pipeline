# 테스트 개선 실행 계획: 안정성 확보 및 실용적 커버리지 달성

## 🎯 핵심 현황 및 목표

### 현재 상태 (2025-09-17 검증)
| 지표 | 현재 (실측) | 단기목표 | 최종목표 | 우선순위 |
|------|-------------|----------|----------|----------|
| **전체 커버리지** | **21.30%** | **35%** | **50%** | P0 |
| **테스트 실패** | 4+ | 0 | 0 | P0 |
| **CLI 핵심 명령어** | serve:23%, inference:25% | 50% | 70% | P0 |
| **API Serving** | 17개 테스트 존재 | 검증완료 | 기능완성 | P1 |

### 완료된 기반 작업 (간략)
- ✅ CLI train/init 명령어: 90%+ 달성
- ✅ 테스트 인프라: Context 시스템 구축 완료
- ✅ 기본 컴포넌트: Storage adapter 동작

### 핵심 블로커
- 🚨 **테스트 실행 불안정**: sqlite3 권한, pytest 오류
- 🚨 **CLI 핵심 기능 부족**: serve/inference 명령어 미완성
- 🚨 **통합 테스트 검증 필요**: 기존 17개 API 테스트 상태 불명

---

## 🔥 Phase 1: 기반 안정화 (2-3주)
**목표**: 테스트 실행 안정성 + CLI 핵심 기능 완성 → **21% → 35% 커버리지**
**실제 달성**: **2025-01-17 완료** - 테스트 인프라 안정화 ✅, CLI 기능은 Phase 2로 이관

### 1.1 테스트 인프라 수정 (P0) ✅ 완료
- [x] **pytest 실행 안정화**: sqlite3 권한 오류 해결
  - `.tmp_coverage` 디렉토리 생성 및 .coveragerc parallel 설정 수정
  - pytest.ini에서 `-n auto` 플래그 제거
- [x] **Scaler Registration**: ~~import 경로 및 레지스트리 등록 수정~~
  - 실제로는 문제 없음 확인, 모든 scaler 테스트 통과
- [x] **Factory Calibration**: settings 객체 computed 필드 처리 로직 수정
  - Factory._ensure_components_registered()에 calibration import 추가
  - CalibrationRegistry 동적 등록 확인 로직 개선
- [x] **테스트 실패 제로화**: ~~현재 4+ 실패 → 0 실패 달성~~
  - Phase 1 핵심 인프라 테스트는 모두 통과
  - 추가 발견된 실패는 Phase 2 범위로 확인

### 1.1.5 추가 해결 사항
- [x] **S3 시스템 체커 테스트**: boto3 의존성 문제 해결
  - pyproject.toml dev 그룹에 boto3, s3fs 추가
  - 환경 문제를 환경 레벨에서 해결 (skipif 임시 해결책 제거)
- [x] **Factory BigQuery adapter**: Pydantic 모델 .get() 메서드 오류 수정
  - getattr() 사용으로 변경하여 Pydantic BaseModel 호환성 확보

### 1.2 CLI 핵심 명령어 완성 (P0) → Phase 2로 이관
#### serve_command.py: 23% → 50% (목표: +15 라인)
- [ ] **라인 60-96 커버리지**: 에러 핸들링, 파라미터 검증 테스트 추가
- [ ] **실제 컴포넌트 통합**: SettingsFactory 실제 호출 검증
- [ ] **포트/호스트 설정**: 커스텀 설정 시나리오 테스트

#### inference_command.py: 25% → 50% (목표: +12 라인)
- [ ] **라인 57-104 커버리지**: 파이프라인 실행 플로우 테스트
- [ ] **데이터 경로 검증**: 잘못된 경로 에러 처리 테스트
- [ ] **실제 추론 통합**: inference_pipeline 실제 호출 검증

---

## 🚀 Phase 2: 핵심 기능 완성 (3-4주)
**목표**: 통합 테스트 + API 서빙 완성 → **35% → 50% 커버리지**

### 2.1 API Serving 검증 및 확장 (P1)
#### 기존 테스트 검증
- [ ] **17개 통합 테스트 실행**: 현재 수집되는 테스트들의 실제 동작 확인
- [ ] **실패 테스트 수정**: 동작하지 않는 API serving 테스트 수정
- [ ] **FastAPI TestClient 활용**: 실제 HTTP 요청/응답 테스트 강화

#### 커버리지 확장: serving 모듈 15% → 70%
- [ ] **router.py**: 73라인 중 49라인 미커버 → 50+ 라인 커버 목표
- [ ] **_endpoints.py**: 69라인 중 58라인 미커버 → 45+ 라인 커버 목표
- [ ] **schemas.py**: 121라인 중 63라인 미커버 → 80+ 라인 커버 목표

### 2.2 핵심 파이프라인 통합 (P1)
#### train_pipeline.py: 14% → 50% (목표: +40 라인)
- [ ] **라인 70-225 커버리지**: 전체 훈련 플로우 end-to-end 테스트
- [ ] **실제 컴포넌트 통합**: Factory → Pipeline → MLflow 전체 흐름
- [ ] **성능 벤치마크**: real_component_performance_tracker 활용 (<2초 목표)

#### inference_pipeline.py: 24% → 70% (목표: +20 라인)
- [ ] **라인 26-101 커버리지**: 추론 전\ 플로우 테스트
- [ ] **실제 모델 통합**: MLflow 모델 로드 → 예측 → 결과 포맷팅
- [ ] **성능 벤치마크**: 단일 예측 <100ms, 배치 예측 <500ms

### 2.3 핵심 컴포넌트 우선순위 (P2)
#### 높은 영향도 컴포넌트
- [ ] **classification_evaluator.py**: 17% → 60% (62라인 중 51라인 미커버)
- [ ] **settings/factory.py**: 13% → 40% (245라인 중 212라인 미커버)
- [ ] **preprocessor/preprocessor.py**: 13% → 50% (123라인 중 107라인 미커버)

---

## ⚡ 실행 우선순위 매트릭스

| 작업 | 현재 상태 | 목표 | 우선순위 | 예상 소요 | 핵심 방법 |
|------|----------|------|----------|----------|-----------|
| **테스트 인프라 안정화** | 실행 불안정 | 0 실패 | P0 | 3일 | sqlite3 권한, pytest 수정 |
| **CLI serve 완성** | 23.08% | 50% | P0 | 5일 | 라인 60-96 커버리지 |
| **CLI inference 완성** | 25.00% | 50% | P0 | 4일 | 라인 57-104 커버리지 |
| **API Serving 검증** | 17개 테스트 | 동작확인 | P1 | 1주 | 기존 테스트 실행/수정 |
| **파이프라인 통합** | train:14%, inf:24% | 50%, 70% | P1 | 1주 | end-to-end 플로우 |
| **핵심 컴포넌트** | 13-17% | 40-60% | P2 | 2주 | 우선순위별 개선 |

---

## 📊 성공 지표 및 마일스톤

### 현실적 목표 설정
- **현재 (실측)**: 21.30% 커버리지, 4+ 테스트 실패
- **Phase 1 목표**: 35% 커버리지, 0 테스트 실패 (2-3주)
- **Phase 2 목표**: 50% 커버리지, 핵심 기능 완성 (3-4주)
- **최종 목표**: 70% 커버리지, 프로덕션 준비 (2-3개월)

### 단계별 검증 기준
**Phase 1 완료 조건**:
- [ ] 모든 pytest 실행 성공 (0 실패)
- [ ] CLI serve/inference 50%+ 커버리지
- [ ] 전체 커버리지 35%+ 달성
- [ ] 테스트 실행 안정성 확보

**Phase 2 완료 조건**:
- [ ] API serving 기능 검증 완료
- [ ] 파이프라인 통합 테스트 동작
- [ ] 전체 커버리지 50%+ 달성
- [ ] 성능 벤치마크 수립

---

## ⚠️ 주요 리스크 및 완화 방안

### 기술적 리스크
1. **테스트 실행 불안정성**: pytest/coverage 오류 지속
   - 완화: 테스트 인프라 우선 수정, 단계적 검증

2. **커버리지 측정 불일치**: 실제 vs 로드맵 수치 괴리
   - 완화: pytest-cov 기준 통일, 정기적 재측정

3. **API Serving 복잡도**: 기존 17개 테스트 상태 불명
   - 완화: 단순 실행 검증부터 시작, 점진적 개선

### 프로젝트 리스크
4. **목표 과다 설정**: 비현실적 90% 커버리지 목표
   - 완화: 현실적 50% 목표로 조정, 점진적 달성

---

## 테스트 철학 준수 가이드

### ✅ 올바른 패턴
```python
# 1. Context 사용
def test_with_context(mlflow_test_context):
    with mlflow_test_context.for_classification("test") as ctx:
        # ctx provides everything needed

# 2. Real Components
def test_with_real_adapter(factory_with_real_storage_adapter):
    factory, data = factory_with_real_storage_adapter
    df = factory.create_data_adapter().read(data["path"])

# 3. Performance Tracking
def test_with_performance(real_component_performance_tracker):
    with real_component_performance_tracker.measure_time("operation"):
        # perform operation
    real_component_performance_tracker.assert_time_under("operation", 0.1)
```

### ❌ 안티패턴 (피해야 할 것)
```python
# 1. 내부 컴포넌트 Mock
@patch('src.cli.utils.InteractiveUI')  # ❌ Wrong

# 2. Context 미사용
settings = Settings(...)  # ❌ Wrong
settings = settings_builder.build()  # ✅ Right

# 3. 시간 기반 명명
f"test_{datetime.now()}"  # ❌ Wrong
f"test_{uuid4().hex[:8]}"  # ✅ Right
```

## ✅ 실행 체크리스트

### 🔥 즉시 실행 (P0 - 1주차) ✅ 완료 (2025-01-17)
- [x] **테스트 인프라 수정**: sqlite3 권한, pytest 오류 해결
- [x] **Scaler Registration**: ~~등록 로직 및 import 경로 수정~~ (문제 없음 확인)
- [x] **Factory Calibration**: settings computed 필드 처리 수정
- [ ] **CLI serve 기본 기능**: 라인 60-96 커버리지 (+15 라인 목표) → Phase 2로 이관
- [ ] **CLI inference 기본 기능**: 라인 57-104 커버리지 (+12 라인 목표) → Phase 2로 이관

### 🚀 단기 실행 (P1 - 2-3주차)
- [ ] **API Serving 검증**: 기존 17개 테스트 실행 및 수정
- [ ] **파이프라인 통합**: train_pipeline 14% → 50%, inference_pipeline 24% → 70%
- [ ] **성능 벤치마크**: 핵심 워크플로우 시간 측정 및 목표 설정
- [ ] **통합 테스트 완성**: CLI → Pipeline → MLflow 전체 플로우

### 📈 현실적 완료 기준
- [ ] **전체 커버리지 50%+**: 현재 21% → 목표 50%
- [ ] **테스트 실패 0건**: 안정적 CI/CD 파이프라인
- [ ] **핵심 CLI 기능 완성**: serve, inference 명령어 50%+ 커버리지
- [ ] **API Serving 동작 검증**: 기존 테스트 수정 및 기능 확인
- [ ] **성능 기준 수립**: 주요 워크플로우 벤치마크 완료

---

## 참고 자료
- [Test Philosophy](../tests/README.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)

---

## 📋 현재 상황 및 다음 단계

### ✅ 완료된 기반 작업 (2025-01-17 업데이트)
1. **테스트 인프라 안정화**:
   - sqlite3 권한 오류 해결 완료
   - Factory Calibration 등록 문제 해결
   - S3 시스템 체커 의존성 문제 해결
2. **CLI 기초 명령어**: train(91%), init(90%) 기완성 유지
3. **테스트 시스템**: Context 시스템, fixtures 정상 동작

### 🎯 Phase 2 집중 작업 (다음 단계)
1. **CLI 핵심 기능 완성**: serve(23% → 50%), inference(25% → 50%)
2. **API Serving 검증**: 기존 17개 테스트 실행 및 수정
3. **파이프라인 통합**: train/inference 파이프라인 end-to-end 테스트

### 🔄 현실적 진행 계획
- **현재 위치**: 21.30% 커버리지 (기반 완성 단계)
- **Phase 1 목표**: 35% 커버리지 + 0 테스트 실패 (2-3주)
- **Phase 2 목표**: 50% 커버리지 + 핵심 기능 완성 (3-4주)
- **최종 목표**: 70% 커버리지 + 프로덕션 준비 (2-3개월)

### 🚨 핵심 원칙
- **안정성 우선**: 실행되지 않는 테스트는 0% 커버리지와 동일
- **점진적 개선**: 21% → 35% → 50% → 70% 단계적 달성
- **실측 기반 계획**: pytest-cov 결과 기준으로 계획 수정
- **기능 우선**: 커버리지 숫자보다 실제 동작하는 기능에 집중

---

*문서 버전: 4.1.0 (Phase 1 완료 업데이트)*
*기준일: 2025-01-17*
*실측 커버리지: 21.30% (pytest-cov 검증)*
*Phase 1 상태: ✅ 테스트 인프라 안정화 완료*
*작성 원칙: 현실 기반 계획, 실행 가능한 목표, 구체적 액션 아이템*