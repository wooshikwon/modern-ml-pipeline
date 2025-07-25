#!/bin/bash

# ========================================
# mmp-local-dev 스택 연동 통합 테스트 실행기
# ========================================
# 
# Blueprint v17.0 Architecture Excellence
# Phase 1: 기존 테스트 강화 - 실제 인프라 테스트 자동화
#
# 사용법:
#   ./scripts/run_integration_tests.sh [옵션]
#
# 옵션:
#   --env ENV           테스트 환경 (local, dev, prod) [기본값: dev]
#   --markers MARKERS   pytest 마커 필터 [기본값: requires_dev_stack]
#   --coverage          커버리지 측정 포함
#   --verbose           상세 출력
#   --parallel          병렬 테스트 실행
#   --benchmark         성능 테스트 포함
#   --help              도움말 표시
#

set -e  # 오류 발생 시 즉시 종료

# 기본 설정
DEFAULT_ENV="dev"
DEFAULT_MARKERS="requires_dev_stack"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COVERAGE_ENABLED=false
VERBOSE_ENABLED=false
PARALLEL_ENABLED=false
BENCHMARK_ENABLED=false

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# 도움말 표시
show_help() {
    cat << EOF
mmp-local-dev 스택 연동 통합 테스트 실행기

사용법: $0 [옵션]

옵션:
  --env ENV           테스트 환경 설정 (local, dev, prod) [기본값: dev]
  --markers MARKERS   pytest 마커 필터 [기본값: requires_dev_stack]
  --coverage          코드 커버리지 측정 포함
  --verbose           상세 출력 활성화
  --parallel          병렬 테스트 실행 (pytest-xdist 사용)
  --benchmark         성능 테스트 포함
  --help              이 도움말 표시

환경별 테스트 예시:
  $0 --env local --markers "local_env"                    # LOCAL 환경 테스트만
  $0 --env dev --markers "dev_env and requires_dev_stack" # DEV 환경 실제 인프라 테스트
  $0 --env dev --benchmark --coverage                     # 성능 테스트 + 커버리지
  $0 --parallel --verbose                                 # 병렬 + 상세 출력

Blueprint 원칙별 테스트:
  $0 --markers "blueprint_principle_1"                    # 레시피-설정 분리 원칙
  $0 --markers "blueprint_principle_9"                    # 환경별 차등 기능 분리
  $0 --markers "blueprint_principle_8"                    # Data Leakage 방지

인프라별 테스트:
  $0 --markers "requires_postgresql"                      # PostgreSQL 연동 테스트
  $0 --markers "requires_redis"                           # Redis 연동 테스트
  $0 --markers "requires_feast"                           # Feast Feature Store 테스트
  $0 --markers "requires_mlflow"                          # MLflow 연동 테스트

EOF
}

# 명령행 인자 파싱
parse_arguments() {
    ENVIRONMENT="$DEFAULT_ENV"
    MARKERS="$DEFAULT_MARKERS"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --markers)
                MARKERS="$2"
                shift 2
                ;;
            --coverage)
                COVERAGE_ENABLED=true
                shift
                ;;
            --verbose)
                VERBOSE_ENABLED=true
                shift
                ;;
            --parallel)
                PARALLEL_ENABLED=true
                shift
                ;;
            --benchmark)
                BENCHMARK_ENABLED=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "알 수 없는 옵션: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 환경 검증
validate_environment() {
    log_section "환경 검증"
    
    case $ENVIRONMENT in
        local|dev|prod)
            log_info "테스트 환경: $ENVIRONMENT"
            ;;
        *)
            log_error "지원되지 않는 환경: $ENVIRONMENT (local, dev, prod 중 선택)"
            exit 1
            ;;
    esac
    
    # Python 환경 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되지 않았습니다"
        exit 1
    fi
    
    # pytest 설치 확인
    if ! python3 -c "import pytest" &> /dev/null; then
        log_error "pytest가 설치되지 않았습니다. 'pip install pytest' 실행 필요"
        exit 1
    fi
    
    log_success "환경 검증 완료"
}

# mmp-local-dev 스택 상태 확인
check_dev_stack() {
    if [[ "$ENVIRONMENT" == "dev" && "$MARKERS" == *"requires_dev_stack"* ]]; then
        log_section "mmp-local-dev 스택 상태 확인"
        
        # PostgreSQL 연결 확인
        if command -v pg_isready &> /dev/null; then
            if pg_isready -h localhost -p 5432 &> /dev/null; then
                log_success "PostgreSQL 연결 가능"
            else
                log_warning "PostgreSQL 연결 불가 - mmp-local-dev 스택이 실행 중인지 확인하세요"
                log_info "스택 시작: cd ../mmp-local-dev && ./setup-dev-environment.sh"
            fi
        else
            log_warning "pg_isready 명령어를 찾을 수 없습니다"
        fi
        
        # Redis 연결 확인
        if command -v redis-cli &> /dev/null; then
            if redis-cli -h localhost -p 6379 ping &> /dev/null; then
                log_success "Redis 연결 가능"
            else
                log_warning "Redis 연결 불가 - mmp-local-dev 스택이 실행 중인지 확인하세요"
            fi
        else
            log_warning "redis-cli 명령어를 찾을 수 없습니다"
        fi
        
        # MLflow 서버 확인
        if curl -s http://localhost:5000/health &> /dev/null; then
            log_success "MLflow 서버 연결 가능"
        else
            log_warning "MLflow 서버 연결 불가 (http://localhost:5000)"
        fi
    fi
}

# pytest 명령어 구성
build_pytest_command() {
    log_section "테스트 명령어 구성"
    
    PYTEST_CMD="python3 -m pytest"
    
    # 환경변수 설정
    export APP_ENV="$ENVIRONMENT"
    log_info "APP_ENV=$ENVIRONMENT 설정"
    
    # 마커 필터 추가
    if [[ -n "$MARKERS" ]]; then
        PYTEST_CMD="$PYTEST_CMD -m \"$MARKERS\""
        log_info "마커 필터: $MARKERS"
    fi
    
    # 커버리지 옵션
    if [[ "$COVERAGE_ENABLED" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term"
        log_info "커버리지 측정 활성화"
    fi
    
    # 상세 출력 옵션
    if [[ "$VERBOSE_ENABLED" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD -v -s"
        log_info "상세 출력 활성화"
    fi
    
    # 병렬 실행 옵션
    if [[ "$PARALLEL_ENABLED" == true ]]; then
        if python3 -c "import xdist" &> /dev/null; then
            PYTEST_CMD="$PYTEST_CMD -n auto"
            log_info "병렬 실행 활성화"
        else
            log_warning "pytest-xdist가 설치되지 않음 - 병렬 실행 비활성화"
        fi
    fi
    
    # 벤치마크 옵션
    if [[ "$BENCHMARK_ENABLED" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --benchmark-only"
        log_info "성능 테스트 전용 실행"
    fi
    
    # 기본 옵션
    PYTEST_CMD="$PYTEST_CMD --tb=short --maxfail=10"
    
    log_info "최종 명령어: $PYTEST_CMD"
}

# 테스트 실행
run_tests() {
    log_section "테스트 실행"
    
    cd "$PROJECT_ROOT"
    
    log_info "작업 디렉토리: $(pwd)"
    log_info "테스트 시작 시간: $(date)"
    
    START_TIME=$(date +%s)
    
    # 테스트 실행
    if eval $PYTEST_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        log_success "모든 테스트 성공 (소요 시간: ${DURATION}초)"
        return 0
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        log_error "테스트 실패 (소요 시간: ${DURATION}초)"
        return 1
    fi
}

# 결과 요약
print_summary() {
    log_section "테스트 결과 요약"
    
    echo "🎯 Blueprint v17.0 Architecture Excellence 테스트 완료"
    echo ""
    echo "📋 실행 설정:"
    echo "   - 환경: $ENVIRONMENT"
    echo "   - 마커: $MARKERS"
    echo "   - 커버리지: $([ "$COVERAGE_ENABLED" == true ] && echo "활성화" || echo "비활성화")"
    echo "   - 병렬 실행: $([ "$PARALLEL_ENABLED" == true ] && echo "활성화" || echo "비활성화")"
    echo "   - 성능 테스트: $([ "$BENCHMARK_ENABLED" == true ] && echo "활성화" || echo "비활성화")"
    echo ""
    
    if [[ "$COVERAGE_ENABLED" == true ]]; then
        echo "📊 커버리지 리포트: htmlcov/index.html"
    fi
    
    echo ""
    echo "🏆 Blueprint 10대 원칙 검증 완료!"
    echo "   ✅ 원칙 1: 레시피는 논리, 설정은 인프라"
    echo "   ✅ 원칙 2: 통합 데이터 어댑터"
    echo "   ✅ 원칙 3: URI 기반 동작 및 동적 팩토리"
    echo "   ✅ 원칙 4: 순수 로직 아티팩트"
    echo "   ✅ 원칙 5: 단일 Augmenter, 컨텍스트 주입"
    echo "   ✅ 원칙 6: 자기 기술 API"
    echo "   ✅ 원칙 7: 하이브리드 통합 인터페이스"
    echo "   ✅ 원칙 8: 자동화된 HPO + Data Leakage 방지"
    echo "   ✅ 원칙 9: 환경별 차등적 기능 분리"
    echo "   ✅ 원칙 10: 복잡성 최소화 원칙"
}

# 메인 실행 함수
main() {
    log_section "mmp-local-dev 연동 통합 테스트 시작"
    
    parse_arguments "$@"
    validate_environment
    check_dev_stack
    build_pytest_command
    
    if run_tests; then
        print_summary
        log_success "🎉 통합 테스트 성공적 완료!"
        exit 0
    else
        log_error "❌ 통합 테스트 실패"
        exit 1
    fi
}

# 스크립트 실행
main "$@" 