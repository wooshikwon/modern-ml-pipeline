# UnifiedConsole 메서드 통합 리팩토링 계획

**작성일**: 2025-01-15
**목적**: UnifiedConsole의 중복 메서드 제거 및 일관된 인터페이스 구축
**원칙**: 하위호환성 없이 완전한 전환 (Clean Migration)

## 📊 현황 분석

### 메서드 사용 현황
| 메서드명 | 사용 빈도 | 주요 사용처 | 상태 |
|---------|-----------|------------|------|
| `log_processing_step` | 51회 | 전체 | ✅ 핵심 |
| `log_milestone` | 24회 | 파이프라인 | ✅ 유지 |
| `log_data_operation` | 24회 | DataHandler | 🔄 통합 대상 |
| `log_phase` | 14회 | 파이프라인 | ✅ 유지 |
| `log_model_operation` | 13회 | Trainer | 🔄 통합 대상 |
| `log_file_operation` | 5회 | StorageAdapter | ❌ 제거 대상 |
| `log_database_operation` | 4회 | SqlAdapter | ❌ 제거 대상 |
| `log_feature_engineering` | 0회 | 없음 | ❌ 즉시 제거 |

### 문제점
1. **기능 중복**: 모든 `log_*_operation` 메서드가 동일한 구조
2. **일관성 부족**: 도메인별로 다른 메서드명 사용
3. **유지보수 어려움**: 새로운 도메인 추가 시 새 메서드 필요

## 🎯 목표 아키텍처

### 최종 메서드 구조
```python
class UnifiedConsole:
    # === 핵심 이벤트 로깅 ===
    def log_milestone(message: str, level: str = "info")  # 중요 이벤트
    def log_phase(phase_name: str, emoji: str = "📝")     # 단계 구분

    # === 통합 작업 로깅 (새로운) ===
    def log_operation(
        operation: str,
        details: str = "",
        operation_type: str = "general",  # general|data|model|file|db
        shape: tuple = None
    )

    # === 에러/경고 ===
    def log_error_with_context(...)
    def log_warning_with_context(...)

    # === 특수 목적 (유지) ===
    def log_validation_result(...)
    def log_connection_status(...)
    def log_pipeline_connection(...)
    def log_performance_guidance(...)

    # === 기타 유틸리티 ===
    def log_periodic(...)
    def log_artifacts_progress(...)
```

## 📝 실행 계획

### Phase 1: 즉시 제거 (Quick Win)
**목표**: 사용 빈도가 낮은 메서드 즉시 제거
**일정**: 즉시 실행

#### 1.1 StorageAdapter 수정
- [ ] `log_file_operation` → `log_processing_step` 전환
- 파일: `src/components/adapter/modules/storage_adapter.py`
- 변경 내용:
  ```python
  # Before
  console.log_file_operation(f"Storage 파일 읽기 시작", uri, f"유형: {file_ext}")

  # After
  console.log_processing_step(
      f"Storage 파일 읽기 시작: {Path(uri).name}",
      f"유형: {file_ext}, Storage 옵션: {len(self.storage_options)}개"
  )
  ```

#### 1.2 불필요 메서드 제거
- [ ] `log_feature_engineering` 메서드 제거 (사용처 없음)
- [ ] `log_component_init` → `log_milestone`로 전환 (2곳)

### Phase 2: 통합 메서드 추가
**목표**: 새로운 통합 인터페이스 구축
**일정**: Phase 1 완료 후

#### 2.1 UnifiedConsole에 통합 메서드 추가
```python
def log_operation(self,
                 operation: str,
                 details: str = "",
                 operation_type: str = "general",
                 shape: tuple = None):
    """
    통합된 작업 로깅 메서드

    Args:
        operation: 작업 설명
        details: 상세 정보
        operation_type: 'general', 'data', 'model', 'file', 'db'
        shape: 데이터 shape (data 타입일 때만)
    """
    emoji_map = {
        'general': '🔄',
        'data': '📊',
        'model': '🤖',
        'file': '📁',
        'db': '🗄️'
    }
    emoji = emoji_map.get(operation_type, '🔄')

    shape_str = f" ({shape[0]} rows, {shape[1]} columns)" if shape else ""
    self.console.print(f"{emoji} {operation}{shape_str}")
    if details:
        self.console.print(f"   [dim]{details}[/dim]")
```

### Phase 3: 전체 마이그레이션
**목표**: 모든 컴포넌트를 새 인터페이스로 전환
**일정**: Phase 2 완료 후

#### 3.1 컴포넌트별 전환 순서
1. [ ] StorageAdapter (이미 Phase 1에서 처리)
2. [ ] SqlAdapter: `log_database_operation` → `log_operation(..., operation_type='db')`
3. [ ] Trainer: `log_model_operation` → `log_operation(..., operation_type='model')`
4. [ ] DataHandler: `log_data_operation` → `log_operation(..., operation_type='data')`
5. [ ] 기타 컴포넌트: `log_processing_step` → `log_operation(..., operation_type='general')`

#### 3.2 마이그레이션 매핑
| 기존 메서드 | 새 메서드 호출 |
|------------|---------------|
| `log_processing_step(op, details)` | `log_operation(op, details, 'general')` |
| `log_data_operation(op, shape, details)` | `log_operation(op, details, 'data', shape)` |
| `log_model_operation(op, info)` | `log_operation(op, info, 'model')` |
| `log_file_operation(op, path, details)` | `log_operation(f"{op}: {Path(path).name}", details, 'file')` |
| `log_database_operation(op, details)` | `log_operation(op, details, 'db')` |

### Phase 4: 정리 및 최종화
**목표**: 레거시 코드 완전 제거
**일정**: Phase 3 완료 후

#### 4.1 레거시 메서드 제거
- [ ] `log_processing_step` 제거
- [ ] `log_data_operation` 제거
- [ ] `log_model_operation` 제거
- [ ] `log_file_operation` 제거
- [ ] `log_database_operation` 제거
- [ ] `log_feature_engineering` 제거
- [ ] `log_component_init` 제거

#### 4.2 문서 업데이트
- [ ] UnifiedConsole 클래스 docstring 업데이트
- [ ] 사용 가이드 작성
- [ ] 마이그레이션 완료 보고서 작성

## 🧪 테스트 계획

### 각 Phase별 검증
1. **Phase 1 후**: StorageAdapter 테스트 통과 확인
   ```bash
   pytest tests/unit/factory/test_component_creation.py::TestFactoryWithRealComponents::test_factory_creates_real_storage_adapter -v
   ```

2. **Phase 2 후**: 통합 메서드 단위 테스트
   ```bash
   pytest tests/unit/utils/core/test_console_manager.py -v
   ```

3. **Phase 3 후**: 전체 컴포넌트 테스트
   ```bash
   pytest tests/unit/ -v
   ```

4. **Phase 4 후**: 전체 테스트 스위트
   ```bash
   pytest tests/ -v
   ```

## 📈 성과 지표

### 정량적 지표
- 메서드 수: 17개 → 10개 (41% 감소)
- 코드 라인: 약 150줄 감소 예상
- 중복 코드: 완전 제거

### 정성적 개선
- ✅ 일관된 인터페이스
- ✅ 명확한 의도 표현 (operation_type)
- ✅ 유지보수성 향상
- ✅ 새 도메인 추가 용이

## ⚠️ 주의사항

1. **하위호환성 없음**: 모든 사용처를 한 번에 변경
2. **테스트 우선**: 각 단계마다 테스트 검증 필수
3. **원자적 커밋**: 각 Phase를 별도 커밋으로 관리

## 🚀 시작하기

```bash
# 1. 현재 테스트 상태 확인
pytest tests/unit/factory/test_component_creation.py -v

# 2. Phase 1 실행
# StorageAdapter 수정...

# 3. 테스트 재실행
pytest tests/unit/factory/test_component_creation.py -v

# 성공 시 다음 Phase 진행
```

---

**Status**: 🔄 Phase 1 진행 중
**Last Updated**: 2025-01-15