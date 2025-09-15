# Console Manager 통합 리팩토링 상세 계획

**작성일**: 2025-01-15
**목적**: console_manager.py의 복잡한 이중 구조를 단일 통합 클래스로 단순화

## 🔍 현재 구조 분석

### 현재 문제점
```
현재: RichConsoleManager (383 lines) + UnifiedConsole (83 lines) = 466 lines
문제:
1. UnifiedConsole이 RichConsoleManager를 래핑하는 불필요한 계층
2. 일부 메서드는 RichConsoleManager에만 존재 (log_processing_step 등)
3. UnifiedConsole이 모든 메서드를 래핑하지 않아 에러 발생
4. 중복된 환경 감지 로직
5. 817줄의 거대한 파일
```

### 사용 패턴 분석
```python
# 현재 사용 패턴
console = get_console(settings)  # UnifiedConsole 반환
console.log_processing_step(...)  # ❌ AttributeError!

# 내부 구조
UnifiedConsole.rich_console = RichConsoleManager()
UnifiedConsole.rich_console.log_processing_step(...)  # 실제 메서드는 여기에
```

## 🎯 목표 아키텍처

### 단일 통합 클래스
```python
class Console:
    """
    통합 콘솔 매니저
    - Rich 출력과 로깅을 통합 관리
    - 환경별 자동 적응 (CI/CD, 터미널, 테스트)
    - 단순하고 일관된 인터페이스
    """

    def __init__(self, settings=None):
        # Rich console
        self.console = RichConsole()

        # Logger integration
        from src.utils.core.logger import logger
        self.logger = logger

        # Environment detection
        self.mode = self._detect_mode(settings)

        # Progress tracking
        self.progress_bars = {}
        self.active_progress = None

    # === 핵심 메서드 (통합) ===
    def log(self,
            message: str,
            level: str = "info",
            details: str = "",
            context: dict = None,
            operation_type: str = None):
        """
        통합 로깅 메서드
        level: info|success|warning|error
        operation_type: general|data|model|file|db (optional)
        """
        # Logger에 기록
        getattr(self.logger, level, self.logger.info)(message)

        # 콘솔 출력 (환경별 적응)
        if self.mode == "rich":
            self._rich_output(message, level, details, operation_type)
        elif self.mode == "plain":
            self._plain_output(message, level, details)
        elif self.mode == "silent":
            pass  # 테스트 환경 등에서 출력 억제
```

### 메서드 통합 매핑
| 기존 메서드들 | 새 통합 메서드 |
|-------------|--------------|
| `log_milestone()` | `log(level='info')` |
| `log_phase()` | `log(level='phase')` |
| `log_processing_step()` | `log(operation_type='general')` |
| `log_data_operation()` | `log(operation_type='data')` |
| `log_model_operation()` | `log(operation_type='model')` |
| `log_file_operation()` | `log(operation_type='file')` |
| `log_database_operation()` | `log(operation_type='db')` |
| `log_error_with_context()` | `log(level='error', context=...)` |
| `log_warning_with_context()` | `log(level='warning', context=...)` |

## 📝 리팩토링 실행 계획

### Step 1: 새 Console 클래스 작성
```python
# src/utils/core/console.py (새 파일)
class Console:
    """단일 통합 콘솔 클래스"""

    def __init__(self, settings=None):
        from rich.console import Console as RichConsole
        self.rich = RichConsole()
        self.mode = self._detect_mode(settings)
        self._init_logger()

    def log(self, message: str, **kwargs):
        """통합 로깅 인터페이스"""
        # 구현...

    # 호환성을 위한 레거시 메서드들 (deprecated)
    def log_processing_step(self, step: str, details: str = ""):
        """@deprecated: use log() instead"""
        self.log(step, operation_type='general', details=details)

    def log_data_operation(self, op: str, shape=None, details=""):
        """@deprecated: use log() instead"""
        self.log(op, operation_type='data', shape=shape, details=details)

    # ... 다른 레거시 메서드들
```

### Step 2: 점진적 마이그레이션
1. **새 파일 생성**: `console.py`에 통합 Console 클래스
2. **호환성 유지**: 레거시 메서드를 deprecated로 표시하되 유지
3. **import 경로 변경**:
   ```python
   # 기존
   from src.utils.core.console_manager import get_console

   # 새로운
   from src.utils.core.console import get_console
   ```
4. **테스트 및 검증**
5. **레거시 파일 제거**: `console_manager.py` 삭제

### Step 3: 사용 코드 업데이트
```python
# Before (복잡)
console = get_console()
console.log_processing_step("작업 시작", "상세 정보")
console.log_data_operation("데이터 로드", shape=(100, 20))

# After (단순)
console = get_console()
console.log("작업 시작", operation_type='general', details="상세 정보")
console.log("데이터 로드", operation_type='data', shape=(100, 20))
```

## 📊 기대 효과

### 정량적 개선
- 코드 라인: 817줄 → 약 300줄 (63% 감소)
- 클래스 수: 2개 → 1개
- 메서드 수: 30개 → 10개 (핵심만 유지)

### 정성적 개선
- ✅ 단순한 구조: 하나의 클래스, 명확한 인터페이스
- ✅ 일관성: 모든 로깅이 하나의 `log()` 메서드로
- ✅ 유지보수성: 새 기능 추가가 쉬움
- ✅ 테스트 용이성: 단일 진입점으로 모킹 간단

## ⚠️ 리스크 및 대응

### 리스크
1. **많은 파일이 console_manager 사용 중**
   - 대응: 호환성 레이어 제공, 점진적 마이그레이션

2. **테스트 깨질 가능성**
   - 대응: 레거시 메서드 유지, deprecated 경고만 추가

3. **기능 누락 가능성**
   - 대응: 철저한 기능 매핑, 테스트 커버리지 확인

## 🚀 실행 우선순위

### 즉시 실행 (Quick Fix)
```python
# UnifiedConsole에 누락된 메서드 추가 (임시 해결책)
class UnifiedConsole:
    def log_processing_step(self, step_name: str, details: str = ""):
        if self.mode in ["rich", "test"]:
            self.rich_console.log_processing_step(step_name, details)
        else:
            print(f"STEP: {step_name}")
```

### 중기 계획 (Clean Solution)
1. 새 Console 클래스 작성
2. 점진적 마이그레이션
3. 레거시 코드 제거

## 📋 체크리스트

- [ ] 현재 console_manager.py 백업
- [ ] 새 console.py 파일 생성
- [ ] Console 클래스 구현
- [ ] 호환성 레이어 추가
- [ ] StorageAdapter 테스트 통과
- [ ] 전체 테스트 실행
- [ ] 문서 업데이트
- [ ] 레거시 파일 제거

---

**추천**: 먼저 Quick Fix로 당장의 테스트를 통과시킨 후, Clean Solution을 진행