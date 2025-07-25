#!/bin/bash
# 🚀 Modern ML Pipeline - Dev Environment Manager
# 
# 이 스크립트는 modern-ml-pipeline 개발자를 위한 편의성 래퍼입니다.
# ../mmp-local-dev 저장소의 개발 환경을 제어하는 브리지 역할을 합니다.

set -e

# --- 색상 및 로깅 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# --- 환경 변수 ---
MMP_LOCAL_DEV_PATH="../mmp-local-dev"

# --- 사전 조건 확인 ---
check_prerequisites() {
    if [ ! -d "$MMP_LOCAL_DEV_PATH" ]; then
        log_error "'$MMP_LOCAL_DEV_PATH' 디렉토리를 찾을 수 없습니다."
        log_info "mmp-local-dev 저장소를 먼저 클론해주세요:"
        log_info "git clone https://github.com/wooshikwon/mmp-local-dev.git $MMP_LOCAL_DEV_PATH"
        exit 1
    fi

    if [ ! -f "$MMP_LOCAL_DEV_PATH/setup.sh" ]; then
        log_error "'$MMP_LOCAL_DEV_PATH/setup.sh' 스크립트를 찾을 수 없습니다."
        log_info "mmp-local-dev 저장소가 올바른지 확인해주세요."
        exit 1
    fi
}

# --- 관리 명령어 ---
start_env() {
    log_step "DEV 환경 시작 중..."
    (cd "$MMP_LOCAL_DEV_PATH" && ./setup.sh)
    log_success "DEV 환경이 성공적으로 시작되었습니다."
}

stop_env() {
    log_step "DEV 환경 중지 중..."
    (cd "$MMP_LOCAL_DEV_PATH" && ./setup.sh --stop)
    log_success "DEV 환경이 성공적으로 중지되었습니다."
}

clean_env() {
    log_step "DEV 환경 완전 삭제 중..."
    (cd "$MMP_LOCAL_DEV_PATH" && ./setup.sh --clean)
    log_success "DEV 환경이 성공적으로 삭제되었습니다."
}

status_env() {
    log_step "DEV 환경 상태 확인 중..."
    (cd "$MMP_LOCAL_DEV_PATH" && ./setup.sh --status)
}

test_env() {
    log_step "DEV 환경 계약 준수 테스트 실행 중..."
    if [ ! -f "$MMP_LOCAL_DEV_PATH/test-integration.py" ]; then
        log_error "'$MMP_LOCAL_DEV_PATH/test-integration.py'를 찾을 수 없습니다."
        exit 1
    fi
    (cd "$MMP_LOCAL_DEV_PATH" && . .venv/bin/activate && python test-integration.py)
    log_success "계약 준수 테스트가 완료되었습니다."
}

# --- 도움말 ---
print_usage() {
    echo "사용법: $0 <command>"
    echo ""
    echo "명령어:"
    echo "  start   - DEV 환경을 시작하거나 재시작합니다."
    echo "  stop    - DEV 환경을 중지합니다."
    echo "  clean   - DEV 환경을 컨테이너, 볼륨 포함하여 완전히 삭제합니다."
    echo "  status  - DEV 환경의 현재 상태를 확인합니다."
    echo "  test    - DEV 환경이 계약을 준수하는지 테스트합니다."
    echo "  help    - 이 도움말 메시지를 표시합니다."
}

# --- 메인 실행 ---
main() {
    check_prerequisites

    case "$1" in
        start)
            start_env
            ;;
        stop)
            stop_env
            ;;
        clean)
            clean_env
            ;;
        status)
            status_env
            ;;
        test)
            test_env
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "알 수 없는 명령어: '$1'"
            print_usage
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    log_error "명령어를 입력해주세요."
    print_usage
    exit 1
fi

main "$@" 