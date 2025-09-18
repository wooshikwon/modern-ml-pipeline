#!/bin/bash
"""
포괄적 테스트 분석 실행 스크립트
사용법:
  ./scripts/run_test_analysis.sh [phase1,phase2,phase3] [verbose]
  
예시:
  ./scripts/run_test_analysis.sh                    # 모든 Phase 실행
  ./scripts/run_test_analysis.sh 1,2               # Phase 1,2만 실행  
  ./scripts/run_test_analysis.sh 1 verbose         # Phase 1만 상세 실행
"""

set -e  # 에러 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🚀 포괄적 테스트 분석 시작${NC}"
echo -e "${BLUE}📂 프로젝트 루트: ${PROJECT_ROOT}${NC}"

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  가상환경이 활성화되지 않았습니다. 활성화를 시도합니다...${NC}"
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
        echo -e "${GREEN}✅ 가상환경 활성화됨${NC}"
    else
        echo -e "${RED}❌ 가상환경을 찾을 수 없습니다. 수동으로 활성화하세요.${NC}"
        exit 1
    fi
fi

# 필요한 패키지 확인
echo -e "${BLUE}🔍 필요한 패키지 확인 중...${NC}"
python -c "import pytest, coverage" 2>/dev/null || {
    echo -e "${RED}❌ pytest 또는 coverage가 설치되지 않았습니다.${NC}"
    echo -e "${YELLOW}💡 다음 명령어로 설치하세요: pip install pytest coverage pytest-cov pytest-xdist${NC}"
    exit 1
}

# 인자 파싱
PHASES=${1:-"1,2,3"}
VERBOSE_FLAG=""
if [[ "$2" == "verbose" ]]; then
    VERBOSE_FLAG="--verbose"
fi

# Phase 배열로 변환
IFS=',' read -ra PHASE_ARRAY <<< "$PHASES"

echo -e "${BLUE}📋 실행 설정:${NC}"
echo -e "   Phases: ${PHASE_ARRAY[*]}"
echo -e "   Verbose: $([ -n "$VERBOSE_FLAG" ] && echo "Yes" || echo "No")"
echo -e "   출력 파일: test_metrics_comprehensive.json"

# 이전 결과 백업
if [ -f "$PROJECT_ROOT/test_metrics_comprehensive.json" ]; then
    BACKUP_FILE="test_metrics_comprehensive_$(date +%Y%m%d_%H%M%S).json.bak"
    mv "$PROJECT_ROOT/test_metrics_comprehensive.json" "$PROJECT_ROOT/$BACKUP_FILE"
    echo -e "${YELLOW}📦 이전 결과를 ${BACKUP_FILE}로 백업했습니다${NC}"
fi

# htmlcov 디렉토리 정리
if [ -d "$PROJECT_ROOT/htmlcov" ]; then
    rm -rf "$PROJECT_ROOT/htmlcov"
    echo -e "${YELLOW}🧹 이전 커버리지 리포트를 정리했습니다${NC}"
fi

# 실행
echo -e "${GREEN}🏃‍♂️ 테스트 실행 시작...${NC}"
echo -e "${BLUE}============================================${NC}"

cd "$PROJECT_ROOT"

# Python 스크립트 실행
python scripts/comprehensive_test_runner.py \
    --phases ${PHASE_ARRAY[*]} \
    --project-root "$PROJECT_ROOT" \
    $VERBOSE_FLAG

EXIT_CODE=$?

echo -e "${BLUE}============================================${NC}"

# 결과 확인
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 테스트 분석이 성공적으로 완료되었습니다!${NC}"
    
    # 결과 파일 확인
    if [ -f "test_metrics_comprehensive.json" ]; then
        echo -e "${GREEN}📊 상세 결과: test_metrics_comprehensive.json${NC}"
        
        # 간단한 요약 출력
        echo -e "${BLUE}📋 빠른 요약:${NC}"
        python -c "
import json
try:
    with open('test_metrics_comprehensive.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    summary = data['execution_summary']
    print(f'   총 소요시간: {summary[\"total_duration\"]}')
    print(f'   총 테스트: {summary[\"total_tests\"]}개')
    print(f'   성공률: {(summary[\"passed\"] / summary[\"total_tests\"] * 100):.1f}%')
    print(f'   실패율: {summary[\"overall_error_rate\"]:.1f}%')
    print(f'   스킵율: {summary[\"overall_skip_rate\"]:.1f}%')
    print(f'   최종 커버리지: {summary[\"final_coverage\"]:.1f}%')
except Exception as e:
    print(f'   요약 파싱 실패: {e}')
"
    fi
    
    # HTML 커버리지 리포트 확인
    if [ -d "htmlcov" ]; then
        echo -e "${GREEN}🌐 HTML 커버리지 리포트가 생성되었습니다${NC}"
        echo -e "${BLUE}   브라우저에서 확인: file://$PROJECT_ROOT/htmlcov/index.html${NC}"
    fi
    
    # 로그 파일 안내
    if [ -f "test_runner.log" ]; then
        echo -e "${GREEN}📝 상세 로그: test_runner.log${NC}"
    fi
    
elif [ $EXIT_CODE -eq 130 ]; then
    echo -e "${YELLOW}⛔ 사용자에 의해 중단되었습니다${NC}"
    
    # 중간 결과 확인
    if [ -f "test_metrics_intermediate.json" ]; then
        echo -e "${YELLOW}📊 중간 결과가 저장되었습니다: test_metrics_intermediate.json${NC}"
    fi
    
else
    echo -e "${RED}❌ 테스트 실행 중 오류가 발생했습니다 (exit code: $EXIT_CODE)${NC}"
    
    # 에러 로그 확인
    if [ -f "test_runner.log" ]; then
        echo -e "${RED}📝 에러 로그를 확인하세요: test_runner.log${NC}"
        echo -e "${YELLOW}마지막 10줄:${NC}"
        tail -10 test_runner.log
    fi
fi

echo -e "${BLUE}📚 추가 분석을 위해 다음 파일들을 확인하세요:${NC}"
echo -e "   - test_metrics_comprehensive.json: 최종 상세 리포트"
echo -e "   - test_runner.log: 실행 로그"
echo -e "   - htmlcov/: HTML 커버리지 리포트"
echo -e "   - test_metrics_intermediate.json: 중간 결과 (중단된 경우)"

exit $EXIT_CODE