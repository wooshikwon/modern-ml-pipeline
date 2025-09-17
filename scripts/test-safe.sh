#!/bin/bash
# 안전한 테스트 실행 스크립트 - OOM 방지

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Coverage 파일 정리 함수
cleanup_coverage() {
    log_info "Coverage 파일 정리 중..."

    # 프로젝트 루트의 coverage 파일들 정리
    rm -f .coverage.*
    rm -f .coverage

    # 임시 coverage 디렉토리 정리
    rm -rf .tmp_coverage

    log_info "Coverage 파일 정리 완료"
}

# Coverage 결과 통합 및 정리 함수
finalize_coverage() {
    log_info "Coverage 결과 통합 중..."

    # 임시 디렉토리 존재 확인
    if [ -d ".tmp_coverage" ]; then
        cd .tmp_coverage 2>/dev/null || return 1

        # Coverage 파일들이 존재하면 combine 실행
        if ls .coverage.* 1> /dev/null 2>&1; then
            if command -v coverage >/dev/null 2>&1; then
                coverage combine 2>/dev/null || true
                # 통합된 파일을 프로젝트 루트로 이동
                if [ -f ".coverage" ]; then
                    mv .coverage ../.coverage 2>/dev/null || true
                fi
            fi
        fi

        cd ..
        # 임시 디렉토리 정리
        rm -rf .tmp_coverage
    fi

    log_info "Coverage 결과 통합 완료"
}

# MLflow 서버 정리 함수
cleanup_mlflow_servers() {
    log_info "MLflow 서버 정리 중..."

    # MLflow 서버 프로세스 종료
    if command -v pkill >/dev/null 2>&1; then
        pkill -f "mlflow.server" 2>/dev/null || true
        pkill -f "uvicorn.*mlflow" 2>/dev/null || true
    fi

    # 잠시 대기 후 강제 종료
    sleep 1
    if command -v pkill >/dev/null 2>&1; then
        pkill -9 -f "mlflow.server" 2>/dev/null || true
        pkill -9 -f "uvicorn.*mlflow" 2>/dev/null || true
    fi

    log_info "MLflow 서버 정리 완료"
}

# 메모리 사용량 체크 함수
check_memory() {
    if command -v free >/dev/null 2>&1; then
        # Linux
        available=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$available" -lt 2048 ]; then
            log_warn "사용 가능한 메모리가 부족합니다: ${available}MB"
            return 1
        fi
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        page_size=$(vm_stat | grep "page size" | awk '{print $8}')
        available_mb=$((free_pages * page_size / 1024 / 1024))
        if [ "$available_mb" -lt 2048 ]; then
            log_warn "사용 가능한 메모리가 부족합니다: ${available_mb}MB"
            return 1
        fi
    fi
    return 0
}

# 테스트 전 정리
pre_test_cleanup() {
    log_info "테스트 전 정리 작업 시작..."

    # MLflow 서버 정리 (메모리 누수 방지)
    cleanup_mlflow_servers

    # Coverage 파일 정리
    cleanup_coverage

    # Python 캐시 정리
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    # HTML 커버리지 리포트 정리
    rm -rf htmlcov 2>/dev/null || true

    log_info "정리 작업 완료"
}

# 테스트 후 정리
post_test_cleanup() {
    log_info "테스트 후 정리 작업 시작..."

    # MLflow 서버 정리 (메모리 누수 방지)
    cleanup_mlflow_servers

    # Coverage 결과 통합 및 정리
    finalize_coverage

    log_info "정리 작업 완료"
}

# 단계별 테스트 실행 함수
run_tests_by_category() {
    local category=$1
    local pattern=$2

    log_info "=== $category 테스트 실행 ==="

    # 메모리 체크
    if ! check_memory; then
        log_error "메모리 부족으로 테스트를 건너뜁니다: $category"
        return 1
    fi

    # Coverage 파일 정리
    cleanup_coverage

    # 테스트 실행
    if timeout 300 uv run pytest $pattern --maxfail=5 --tb=short; then
        log_info "$category 테스트 성공"
        post_test_cleanup
        return 0
    else
        log_error "$category 테스트 실패"
        post_test_cleanup
        return 1
    fi
}

# 메인 실행 로직
main() {
    log_info "안전한 테스트 실행 시작..."

    # 사전 정리
    pre_test_cleanup

    # 인수가 있으면 특정 테스트만 실행
    if [ $# -gt 0 ]; then
        log_info "특정 테스트 실행: $*"
        if ! check_memory; then
            log_error "메모리 부족으로 테스트를 중단합니다"
            exit 1
        fi
        cleanup_coverage
        timeout 300 uv run pytest "$@" --maxfail=5 --tb=short
        post_test_cleanup
        exit $?
    fi

    # 카테고리별 순차 실행
    local categories=(
        "Unit Tests:tests/unit"
        "Integration Tests:tests/integration"
        "E2E Tests:tests/e2e"
    )

    local failed_categories=()

    for category_info in "${categories[@]}"; do
        local category=$(echo $category_info | cut -d: -f1)
        local pattern=$(echo $category_info | cut -d: -f2)

        if ! run_tests_by_category "$category" "$pattern"; then
            failed_categories+=("$category")
        fi

        # 카테고리 간 휴식 시간
        sleep 2
    done

    # 결과 보고
    if [ ${#failed_categories[@]} -eq 0 ]; then
        log_info "모든 테스트 카테고리가 성공했습니다!"
        exit 0
    else
        log_error "실패한 테스트 카테고리: ${failed_categories[*]}"
        exit 1
    fi
}

# 스크립트 종료 시 정리
trap post_test_cleanup EXIT

# 메인 실행
main "$@"