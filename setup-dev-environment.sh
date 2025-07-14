#!/bin/bash

# 🚀 Modern ML Pipeline - Development Environment Setup
# Blueprint v17.0: 원스톱 개발환경 자동 설치

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "🚀 Modern ML Pipeline - Development Environment Setup"
    echo "Blueprint v17.0: \"완전한 실험실\" 원스톱 설치"
    echo "=================================================================="
    echo -e "${NC}"
}

# 개발환경 저장소 정보
DEV_ENV_REPO="https://github.com/your-org/mmp-local-dev.git"
DEV_ENV_DIR="../mmp-local-dev"

# 개발환경 다운로드 및 설치
setup_dev_environment() {
    log_step "개발환경 저장소 확인 중..."
    
    if [ -d "$DEV_ENV_DIR" ]; then
        log_info "개발환경이 이미 존재합니다: $DEV_ENV_DIR"
        read -p "기존 환경을 업데이트하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_step "기존 개발환경 업데이트 중..."
            cd "$DEV_ENV_DIR"
            git pull origin main
            cd - > /dev/null
        fi
    else
        log_step "개발환경 저장소 복제 중..."
        git clone "$DEV_ENV_REPO" "$DEV_ENV_DIR"
        log_success "개발환경 저장소 복제 완료"
    fi
}

# 개발환경 자동 설치
install_dev_environment() {
    log_step "개발환경 자동 설치 시작..."
    
    if [ ! -f "$DEV_ENV_DIR/setup.sh" ]; then
        log_error "개발환경 설치 스크립트를 찾을 수 없습니다: $DEV_ENV_DIR/setup.sh"
        exit 1
    fi
    
    cd "$DEV_ENV_DIR"
    chmod +x setup.sh
    
    log_info "Blueprint v17.0 DEV 환경 설치 중..."
    ./setup.sh
    
    cd - > /dev/null
    log_success "개발환경 설치 완료"
}

# 환경 연동 검증
verify_integration() {
    log_step "ML Pipeline과 개발환경 연동 검증 중..."
    
    if [ -f "$DEV_ENV_DIR/test-environment.sh" ]; then
        cd "$DEV_ENV_DIR"
        chmod +x test-environment.sh
        
        log_info "통합 테스트 실행 중..."
        if ./test-environment.sh; then
            log_success "개발환경 연동 검증 완료"
        else
            log_warn "일부 테스트가 실패했지만 계속 진행합니다"
        fi
        
        cd - > /dev/null
    else
        log_warn "통합 테스트 스크립트를 찾을 수 없습니다"
    fi
}

# ML Pipeline 의존성 설치
install_ml_pipeline_deps() {
    log_step "ML Pipeline 의존성 설치 중..."
    
    # Python 환경 확인
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        log_error "Python을 찾을 수 없습니다"
        exit 1
    fi
    
    # 가상환경 생성 (선택적)
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        read -p "Python 가상환경을 생성하시겠습니까? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log_info "Python 가상환경 생성 중..."
            $PYTHON_CMD -m venv .venv
            log_success "가상환경 생성 완료: .venv"
            log_info "활성화: source .venv/bin/activate"
        fi
    fi
    
    # 의존성 설치
    if [ -f "requirements.lock" ]; then
        log_info "ML Pipeline 의존성 설치 중..."
        $PYTHON_CMD -m pip install -r requirements.lock
        log_success "의존성 설치 완료"
    elif [ -f "pyproject.toml" ]; then
        log_info "pyproject.toml 기반 설치 중..."
        $PYTHON_CMD -m pip install -e .
        log_success "프로젝트 설치 완료"
    fi
}

# 사용법 안내
print_usage_guide() {
    echo ""
    echo -e "${GREEN}=================================================================="
    echo "🎉 개발환경 설정이 완료되었습니다!"
    echo "=================================================================="
    echo -e "${NC}"
    
    echo "📱 서비스 접속 정보:"
    echo "   🗄️  PostgreSQL:         localhost:5432"
    echo "   🔴 Redis:              localhost:6379"
    echo "   📊 MLflow:             http://localhost:5000"
    echo "   🐘 pgAdmin:            http://localhost:8082"
    echo "   🔧 Redis Commander:    http://localhost:8081"
    echo ""
    
    echo "🚀 첫 번째 실험 실행:"
    echo "   # 가상환경 활성화 (생성한 경우)"
    echo "   source .venv/bin/activate"
    echo ""
    echo "   # DEV 환경으로 첫 번째 학습"
    echo "   APP_ENV=dev python main.py train --recipe-file models/classification/random_forest_classifier"
    echo ""
    echo "   # 실험 결과 확인"
    echo "   open http://localhost:5000  # MLflow UI"
    echo ""
    
    echo "🛠️ 개발환경 관리:"
    echo "   # 환경 상태 확인"
    echo "   cd $DEV_ENV_DIR && ./setup.sh --status"
    echo ""
    echo "   # 환경 중지"
    echo "   cd $DEV_ENV_DIR && docker-compose down"
    echo ""
    echo "   # 환경 재시작"
    echo "   cd $DEV_ENV_DIR && docker-compose up -d"
    echo ""
    
    echo "📚 문서:"
    echo "   - 개발환경 전체 가이드: $DEV_ENV_DIR/README.md"
    echo "   - Blueprint v17.0 문서: blueprint.md"
    echo "   - Feature Store 사용법: feature_store_contract.md"
    echo ""
}

# 메인 실행
main() {
    print_banner
    
    # 현재 디렉토리 확인
    if [ ! -f "blueprint.md" ] || [ ! -f "main.py" ]; then
        log_error "이 스크립트는 modern-ml-pipeline 프로젝트 루트에서 실행해야 합니다"
        exit 1
    fi
    
    setup_dev_environment
    install_dev_environment
    verify_integration
    install_ml_pipeline_deps
    print_usage_guide
    
    log_success "모든 설정이 완료되었습니다! Blueprint v17.0 개발환경을 즐겨보세요 🚀"
}

# 옵션 처리
case "${1:-}" in
    --dev-only)
        print_banner
        log_info "개발환경만 설치합니다..."
        setup_dev_environment
        install_dev_environment
        verify_integration
        log_success "개발환경 설치 완료"
        exit 0
        ;;
    --ml-only)
        print_banner
        log_info "ML Pipeline 의존성만 설치합니다..."
        install_ml_pipeline_deps
        log_success "ML Pipeline 의존성 설치 완료"
        exit 0
        ;;
    --help)
        echo "Modern ML Pipeline - Development Environment Setup"
        echo ""
        echo "사용법: $0 [옵션]"
        echo ""
        echo "옵션:"
        echo "  (없음)        전체 개발환경 설정 (권장)"
        echo "  --dev-only    개발환경(DB 스택)만 설치"
        echo "  --ml-only     ML Pipeline 의존성만 설치"
        echo "  --help        도움말 표시"
        echo ""
        echo "이 스크립트는 다음을 자동으로 설정합니다:"
        echo "  1. Blueprint v17.0 DEV 환경 다운로드 및 설치"
        echo "  2. PostgreSQL + Redis + MLflow + Feature Store"
        echo "  3. ML Pipeline 의존성 설치"
        echo "  4. 환경 연동 검증"
        echo ""
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "알 수 없는 옵션: $1"
        echo "도움말: $0 --help"
        exit 1
        ;;
esac 