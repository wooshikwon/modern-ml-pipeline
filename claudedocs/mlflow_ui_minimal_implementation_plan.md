# MLflow UI Helper 최소화 구현 계획

## 개요

모델 학습 완료 후 사용자에게 MLflow 웹 UI 접속 정보를 콘솔에 표시하는 최소화된 기능을 구현합니다. 
불필요한 기능들(QR 코드, 브라우저 자동 실행, CLI 플래그, 설정 추가)을 제거하고, 
순수하게 접속 정보만 제공하는 심플한 구현을 목표로 합니다.

## 제거할 기능들

### 1. QR 코드 관련 기능
- `generate_qr_code()` 메서드 완전 제거
- `qrcode` 라이브러리 의존성 제거 
- `ui-extras` 패키지 그룹 제거
- QR 코드 관련 모든 테스트 제거

### 2. 브라우저 자동 실행 기능
- `open_in_browser()` 메서드 완전 제거
- `webbrowser` 모듈 사용 제거
- 브라우저 자동 실행 관련 모든 테스트 제거

### 3. CLI 플래그 추가 제거
- `--open-browser` 플래그 제거
- `--show-qr` 플래그 제거
- `train_command.py`에서 MLflow UI 설정 부분 제거

### 4. 설정 템플릿 추가 제거
- `config.yaml.j2`에서 `mlflow_ui` 섹션 제거
- 환경 변수 관련 설정 제거

## 유지할 핵심 기능들

### 1. MLflowUIHelper 클래스 (간소화)
```python
class MLflowUIHelper:
    def __init__(self, tracking_uri: str, console: Optional[RichConsoleManager] = None)
    def _is_local_uri(self) -> bool                    # 로컬 URI 감지
    def _is_remote_uri(self) -> bool                   # 원격 URI 감지  
    def get_ui_url(self, run_id: str, experiment_id: str) -> str  # URL 생성
    def start_mlflow_server(self, port: int = 5000) -> bool      # 로컬 서버 시작
    def stop_mlflow_server(self)                       # 서버 중지
    def get_network_addresses(self) -> Dict[str, str]  # 네트워크 주소 조회
    def display_access_info(self, run_id: str, experiment_id: str, experiment_name: str)  # 접속 정보 표시
```

### 2. MLflowRunSummary 클래스 (변경 없음)
```python
class MLflowRunSummary:
    def __init__(self, console: Optional[RichConsoleManager] = None)
    def display_run_summary(self, run_id: str, metrics: Dict, params: Dict, artifacts: list = None)
```

### 3. 파이프라인 통합 (간소화)
```python
def _display_mlflow_ui_info(
    settings: Settings,
    run_id: str, 
    run: Any,
    metrics: Dict[str, Any],
    console: RichConsoleManager
):
    """MLflow UI 접속 정보 표시 (간소화된 버전)"""
    # QR, 브라우저 자동실행 관련 파라미터 모두 제거
    # 순수하게 접속 정보와 실행 요약만 표시
```

## 파일별 변경 사항

### 1. src/utils/mlflow/ui_helper.py
**제거할 메서드:**
- `generate_qr_code()` - 완전 제거
- `open_in_browser()` - 완전 제거

**수정할 메서드:**
- `display_access_info()` - `auto_open`, `show_qr` 파라미터 제거

**제거할 import:**
- `import qrcode` 관련
- `import webbrowser` 

### 2. src/pipelines/train_pipeline.py  
**수정할 함수:**
- `_display_mlflow_ui_info()` - 파라미터 간소화, settings 의존성 제거
- 함수 시그니처 단순화

### 3. src/cli/commands/train_command.py
**제거할 부분:**
- `--open-browser` 파라미터
- `--show-qr` 파라미터  
- MLflow UI 설정 주입 코드

### 4. src/cli/templates/configs/config.yaml.j2
**제거할 섹션:**
- `mlflow_ui:` 전체 섹션

### 5. pyproject.toml
**제거할 부분:**
- `ui-extras` 패키지 그룹
- `qrcode>=7.4.0` 의존성

### 6. tests/unit/utils/mlflow/test_ui_helper.py
**제거할 테스트:**
- `test_generate_qr_code()`
- `test_generate_qr_code_import_error()` 
- `test_open_in_browser()`

**수정할 테스트:**
- `test_display_access_info()` - 파라미터 제거된 버전으로 수정

## 구현 순서

### Phase 1: 기능 제거 (위험도: 낮음)
1. **QR 코드 기능 제거**
   - `ui_helper.py`에서 `generate_qr_code()` 메서드 제거
   - 관련 import 제거
   
2. **브라우저 자동 실행 기능 제거** 
   - `open_in_browser()` 메서드 제거
   - `webbrowser` import 제거

### Phase 2: 클래스 간소화 (위험도: 중간)
3. **MLflowUIHelper 클래스 정리**
   - `display_access_info()` 메서드 파라미터 간소화
   - 제거된 기능 관련 코드 정리

### Phase 3: 파이프라인 통합 간소화 (위험도: 중간)  
4. **train_pipeline.py 간소화**
   - `_display_mlflow_ui_info()` 함수 파라미터 제거
   - settings 의존성 제거

### Phase 4: CLI/설정 복원 (위험도: 낮음)
5. **CLI 명령어 복원**
   - `train_command.py`에서 추가된 플래그 제거
   - MLflow UI 설정 주입 코드 제거

6. **설정 템플릿 복원**
   - `config.yaml.j2`에서 `mlflow_ui` 섹션 제거

7. **의존성 정리**
   - `pyproject.toml`에서 `ui-extras` 제거

### Phase 5: 테스트 정리 (위험도: 낮음)
8. **테스트 코드 업데이트** 
   - 제거된 기능 테스트 삭제
   - 수정된 메서드 테스트 업데이트

### Phase 6: 검증 (위험도: 낮음)
9. **통합 테스트**
   - 단위 테스트 실행
   - 실제 학습 파이프라인 실행 테스트

## 예상 결과물

### 최종 동작 방식
1. 사용자가 학습 명령 실행: `mmp train -r recipe.yaml -c config.yaml -d data.csv`
2. 학습 완료 후 자동으로 MLflow UI 정보 표시:
   ```
   📊 Run Summary: run_abc123...
   ┌─────────┬──────────────────┐
   │ Metrics │ • accuracy: 0.95 │
   │         │ • loss: 0.123    │  
   └─────────┴──────────────────┘
   
   🔗 MLflow UI Access
   ┌─────────────┬─────────────────────────────────────┐
   │ Dashboard   │ http://localhost:5000               │
   │ Experiment  │ http://localhost:5000/#/exp/1       │
   │ This Run    │ http://localhost:5000/#/exp/1/run/abc │
   │ Remote      │ http://192.168.1.100:5000          │
   └─────────────┴─────────────────────────────────────┘
   
   ℹ️ Instructions
   • MLflow server is running locally
   • Access the UI at: http://localhost:5000
   • Press Ctrl+C to stop the server
   ```

### 변경 통계 (예상)
- **MLflowUIHelper**: 7개 메서드 → 5개 메서드 (2개 제거)
- **테스트 개수**: 19개 → 15개 (4개 제거) 
- **CLI 파라미터**: 2개 제거 (`--open-browser`, `--show-qr`)
- **의존성**: `qrcode` 패키지 제거
- **설정 항목**: `mlflow_ui` 섹션 제거

## 검증 방법

### 1. 단위 테스트
```bash
python -m pytest tests/unit/utils/mlflow/test_ui_helper.py -v
```

### 2. 통합 테스트  
```bash
# 실제 학습 실행으로 MLflow UI 정보 표시 확인
mmp train -r recipes/test.yaml -c configs/local.yaml -d data/test.csv
```

### 3. 기능 검증 체크리스트
- [ ] 학습 완료 후 MLflow UI URL 표시
- [ ] 로컬 MLflow 서버 자동 시작  
- [ ] 원격 접속 주소 표시
- [ ] 실행 메트릭/파라미터 요약 표시
- [ ] QR 코드 관련 기능 완전 제거 확인
- [ ] 브라우저 자동 실행 기능 완전 제거 확인
- [ ] CLI 플래그 제거 확인
- [ ] 설정 섹션 제거 확인

## 구현 완료 기준

1. **기능 요구사항 충족**: 학습 후 MLflow UI 접속 정보만 콘솔에 표시
2. **불필요 기능 제거**: QR 코드, 브라우저 자동실행 관련 코드 완전 제거
3. **테스트 통과**: 수정된 테스트가 모두 통과
4. **실제 동작 확인**: 실제 학습 파이프라인에서 정상 동작
5. **코드 품질**: 불필요한 import나 주석 없이 깔끔한 코드

---

**작성일**: 2025-01-15  
**담당자**: Claude Code Assistant  
**상태**: 계획 완료, 구현 대기