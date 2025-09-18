# 포괄적 테스트 실행 도구

ML 파이프라인 프로젝트의 방대한 테스트를 16개 그룹으로 체계적으로 분석하는 도구입니다.

## 🚀 빠른 시작

### 모든 테스트 실행
```bash
# 간단 실행 (모든 Phase)
./scripts/run_test_analysis.sh

# 상세 로그와 함께 실행
./scripts/run_test_analysis.sh 1,2,3 verbose
```

### 특정 Phase만 실행
```bash
# Phase 1만 (빠른 단위 테스트)
./scripts/run_test_analysis.sh 1

# Phase 2만 (중간 단위 테스트)  
./scripts/run_test_analysis.sh 2

# Phase 1,2만
./scripts/run_test_analysis.sh 1,2
```

### Python 직접 실행
```bash
# 기본 실행
python scripts/comprehensive_test_runner.py

# 특정 Phase 실행
python scripts/comprehensive_test_runner.py --phases 1 2

# 상세 로그
python scripts/comprehensive_test_runner.py --verbose

# 출력 파일 지정
python scripts/comprehensive_test_runner.py --output my_results.json
```

## 📊 결과 파일

### `test_metrics_comprehensive.json`
최종 종합 리포트 (JSON 형식)
- 실행 요약 (시간, 테스트 수, 성공/실패/스킵율)
- Phase별 상세 결과
- 그룹별 상세 결과  
- 커버리지 진행 과정
- 성능 메트릭

### `htmlcov/` 디렉토리
HTML 커버리지 리포트
- `htmlcov/index.html`: 전체 커버리지 요약
- `htmlcov/group{N}/`: 각 그룹별 커버리지 리포트

### `test_runner.log`
상세 실행 로그
- 타임스탬프별 진행 상황
- 에러 메시지 및 경고
- 각 그룹별 실행 결과

### `test_metrics_intermediate.json`
중간 결과 (중단된 경우)
- 완료된 그룹까지의 결과
- 현재 커버리지 상황

## 🔧 16개 테스트 그룹 구성

### Phase 1: 빠른 단위 테스트 (병렬 실행 가능)
1. **CLI 핵심 기능** - CLI 명령어, 헬프, 설정
2. **CLI 유틸리티** - 설정 빌더, 템플릿 엔진
3. **설정 및 팩토리** - 설정 로딩, 컴포넌트 생성
4. **기본 어댑터 및 페처** - SQL 어댑터, 스토리지 어댑터
5. **유틸리티 핵심** - 로거, 환경 체크, 재현성

### Phase 2: 중간 단위 테스트 (순차 실행)
6. **전처리 컴포넌트** - 스케일러, 인코더, 결측치 처리
7. **데이터 핸들러** - 분류, 회귀, 시계열 데이터 핸들러
8. **모델 컴포넌트** - PyTorch, scikit-learn, 커스텀 모델
9. **평가자 및 캘리브레이션** - 평가 메트릭, 확률 캘리브레이션
10. **트레이너 및 최적화** - 옵티마이저 컴포넌트
11. **파이프라인** - 훈련/추론 파이프라인
12. **서빙** - FastAPI 서빙 컴포넌트
13. **유틸리티 통합** - MLFlow, 외부 서비스 통합

### Phase 3: 통합 및 E2E 테스트 (격리 실행)
14. **데이터베이스 및 MLFlow 통합** - DB 연결, MLFlow 트래킹
15. **컴포넌트 상호작용 및 오류 전파** - 시스템 전체 상호작용
16. **파이프라인 통합 및 서빙** - E2E 워크플로우

## ⚙️ 실행 옵션

### 병렬 처리
- **Phase 1**: `-n auto` (모든 CPU 코어 활용)
- **Phase 2**: `-n 2` 또는 순차 실행 (메모리 관리)
- **Phase 3**: `-n 1` (격리 실행)

### 타임아웃
- **일반 테스트**: 120초
- **모델/GPU 테스트**: 300초
- **통합 테스트**: 600-1200초

### 에러 처리
- **Phase 1**: `--maxfail=15` (관대)
- **Phase 2**: `--maxfail=10` (중간)
- **Phase 3**: `--maxfail=2` (엄격)

## 🎯 메트릭 분석

### 수집 메트릭
- **테스트 커버리지**: 점증적 코드 커버리지
- **에러율**: (실패 테스트 / 전체 테스트) × 100
- **스킵율**: (스킵된 테스트 / 전체 테스트) × 100  
- **실행 속도**: 각 그룹별 소요 시간

### 성능 분석
- 가장 빠른/느린 그룹 식별
- 평균 테스트 실행 시간
- Phase별 소요 시간 분석

## 🐛 문제 해결

### 메모리 부족
```bash
# Phase 2에서 순차 실행 강제
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
./scripts/run_test_analysis.sh 2
```

### 특정 그룹만 실행하고 싶은 경우
```bash
# Python 스크립트에서 원하는 그룹만 활성화
python -c "
import sys
sys.path.append('scripts')
from comprehensive_test_runner import ComprehensiveTestRunner
runner = ComprehensiveTestRunner(Path.cwd())
result = runner.run_group(runner.test_groups[0])  # Group 1만 실행
print(result)
"
```

### 커버리지 리포트 문제
```bash
# 수동 커버리지 통합
coverage combine
coverage html -d htmlcov/manual_report
coverage report --show-missing
```

## 📋 예상 실행 시간

- **Phase 1**: 2-5분 (병렬)
- **Phase 2**: 15-25분 (순차)
- **Phase 3**: 10-20분 (격리)
- **총합**: 27-50분 (환경별 변동)

## 🔍 로그 레벨

### Console 출력
- `INFO`: 주요 진행 상황
- `WARNING`: 경고 사항
- `ERROR`: 심각한 오류

### 파일 로그 (`test_runner.log`)
- `DEBUG`: 상세 디버그 정보
- 모든 pytest 출력 캡처
- 에러 스택 트레이스

## 💡 활용 팁

### 개발 중 빠른 피드백
```bash
# Phase 1만 빠르게 실행
./scripts/run_test_analysis.sh 1 verbose
```

### CI/CD 통합
```bash
# 실패 시 즉시 중단
python scripts/comprehensive_test_runner.py --phases 1 2 3 || exit 1
```

### 커버리지 트래킹
```bash
# 정기적으로 실행하여 커버리지 변화 추적
./scripts/run_test_analysis.sh
# test_metrics_comprehensive.json의 final_coverage 값 모니터링
```