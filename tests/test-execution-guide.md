## **범용 Python 테스트 실행 가이드**

이 문서는 `pytest`를 사용하여 Python 프로젝트의 테스트를 효율적으로 실행, 측정 및 관리하기 위한 표준 방법을 제공합니다. 테스트 아키텍처나 작성법이 아닌, **'어떻게 실행하고 관리할 것인가'**에 집중합니다.

### ## 1. 핵심 실행 명령어 🚀

테스트를 실행하는 가장 기본적인 명령어들입니다.

#### **기본 실행**

  - **전체 테스트 실행**: `pytest` 또는 `pytest tests/`
  - **특정 디렉토리 실행**: `pytest tests/unit/`
  - **특정 파일 실행**: `pytest tests/unit/test_module.py`
  - **특정 테스트 함수 실행**: `pytest tests/unit/test_module.py::test_specific_function`

#### **프로젝트 표준 실행(메트릭 수집 포함)**

  - 그룹 분리 실행 + 커버리지 + 메트릭 집계(표준):
    ```bash
    python3 scripts/run_tests_split.py
    ```
  - 병렬 워커 조절(기본은 안정성 우선으로 각 그룹 1 워커):
    ```bash
    UNIT_WORKERS=1 INTEGRATION_WORKERS=1 E2E_WORKERS=1 python3 scripts/run_tests_split.py
    ```
  - 산출물(`reports/`):
    - `pytest.unit.json`, `pytest.integration.json`, `pytest.e2e.json`
    - `coverage.unit.xml`, `coverage.integration.xml`, `coverage.e2e.xml`
    - `metrics.summary.json`

#### **효율적인 실행**

  - **키워드로 필터링**: 이름에 'login'이 포함된 모든 테스트 실행
    ```bash
    pytest -k "login"
    ```
  - **병렬 실행 (속도 향상)**: `pytest-xdist` 플러그인 필요
    ```bash
    # CPU 코어 수에 맞춰 자동으로 병렬 실행
    # 본 프로젝트 기본값은 안정성 우선으로 -n=1이며, 위 환경 변수로 그룹별 조절
    pytest -n auto
    ```
  - **유용한 옵션**:
      - `-v`: 더 상세한 결과 출력
      - `-x`: 첫 실패 시 즉시 테스트 중단

-----

### ## 2. 테스트 커버리지 측정 📊

코드의 어느 부분이 테스트되었는지 정량적으로 확인하여 테스트의 사각지대를 찾아냅니다.

#### **측정 원칙: 점 표기법(Dot Notation) 사용**

커버리지 측정 대상은 파일 경로가 아닌 **Python 모듈 경로**로 지정해야 정확히 측정됩니다.

  - ✅ **올바른 방법**: `--cov=src.my_app`
  - ❌ **잘못된 방법**: `--cov=src/my_app`

#### **표준 측정 명령어**

  - **전체 프로젝트 커버리지 측정**: 터미널에 미실행 라인까지 표시
    ```bash
    pytest --cov=src --cov-report=term-missing
    ```
  - **HTML 리포트 생성**: 시각적으로 커버리지를 분석할 수 있는 `htmlcov/` 디렉토리 생성
    ```bash
    pytest --cov=src --cov-report=html
    ```

-----

### ## 3. CI/CD 연동 및 자동화 🤖

CI/CD 파이프라인에 테스트 실행 및 커버리지 검증을 통합하여 코드 품질을 자동으로 유지합니다.

#### **최소 커버리지 강제**

`pyproject.toml` (또는 `pytest.ini`)에 최소 커버리지 기준을 설정하여, 기준 미달 시 빌드가 실패하도록 합니다.

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = """
    --cov=src
    --cov-report=term-missing
    --cov-fail-under=80
"""
```

#### **단계적 실행 전략**

피드백 속도와 안정성을 고려하여 파이프라인을 단계적으로 구성합니다.

1.  **빠른 검증 (모든 PR)**: 정적 분석과 단위 테스트를 병렬로 실행하여 빠르게 피드백을 받습니다.
    ```bash
    # 예시: GitHub Actions
    - name: Run Unit Tests
      run: pytest tests/unit -n auto --maxfail=2
    ```
2.  **심층 검증 (주요 브랜치 Merge 전)**: 통합 테스트를 실행하여 컴포넌트 간의 상호작용을 검증합니다.
    ```bash
    - name: Run Integration Tests
      run: pytest tests/integration
    ```
3.  **최종 검증 (배포 전)**: 전체 테스트를 실행하고 최종 커버리지 리포트를 생성합니다.
    ```bash
    - name: Run All Tests & Check Coverage
      run: pytest --cov=src --cov-fail-under=80
    ```

-----

### ## 4. 일반적인 실행 문제 해결 🛠️

  - **문제: 테스트가 로컬에서는 성공하지만 CI 환경에서 실패합니다.**

      - **원인**: 테스트 간 격리가 부족하여 발생하는 상태 오염 또는 경합 조건(race condition)일 가능성이 높습니다. 특히 병렬 실행 시 자주 발생합니다.
      - **해결**: 테스트 아키텍처 가이드의 **테스트 격리 원칙**을 따랐는지 확인하세요. 각 테스트가 독립적인 자원(임시 디렉토리, 고유한 DB 등)을 사용하는지 점검해야 합니다.

  - **문제: 테스트 실행 속도가 너무 느립니다.**

      - **원인**: 특정 테스트의 비효율적인 로직 또는 불필요하게 무거운 환경 설정.
      - **해결**: `pytest -n auto`로 병렬 실행을 우선 적용해 보세요. 그래도 느리다면 `pytest-profiling`과 같은 도구로 병목 현상이 발생하는 테스트를 찾아 최적화합니다.

  - **문제: 커버리지가 0%로 측정됩니다.**

      - **원인**: `cov` 경로를 점 표기법이 아닌 슬래시(`/`) 표기법으로 잘못 지정했거나, `src` 폴더가 Python 경로에 포함되지 않은 경우입니다.
      - **해결**: 측정 명령어가 `--cov=src.my_app`과 같이 **점 표기법**을 올바르게 사용했는지 확인하세요.

-----

### 추가: 프로젝트 특화 실행 정책

- 타임아웃: 기본 `--timeout=60` (테스트 단위)
- 서버 테스트 직렬화: `@pytest.mark.server` + 파일락 기반 `server_serial_execution` 적용
- 프롬프트 소음 억제: `MMP_QUIET_PROMPTS=1`
- 전역 프로세스 강제 종료: `MMP_ENABLE_GLOBAL_KILL=0` (필요 시 `1`로 설정)
- MLflow 격리: 그룹/테스트별 `file://` 기반 임시 디렉토리 사용