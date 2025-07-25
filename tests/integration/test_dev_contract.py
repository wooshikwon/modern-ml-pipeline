"""
"코드로서의 계약" 소비자 측 검증 테스트

이 테스트는 modern-ml-pipeline이 의존하는 mmp-local-dev 인프라가
우리가 기대하는 계약을 준수하는지 자동으로 검증합니다.
"""

import pytest
import yaml
import socket
from pathlib import Path

# --- 경로 설정 ---
# 이 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
# tests/integration/test_dev_contract.py -> modern-ml-pipeline/
ROOT_DIR = Path(__file__).parent.parent.parent
MMP_LOCAL_DEV_PATH = ROOT_DIR.parent / "mmp-local-dev"

# --- 계약 파일 경로 ---
EXPECTED_CONTRACT_PATH = ROOT_DIR / "tests" / "integration" / "expected-dev-contract.yml"
ACTUAL_CONTRACT_PATH = MMP_LOCAL_DEV_PATH / "dev-contract.yml"

# --- 테스트 설정 ---
# mmp-local-dev 스택이 필요한 통합 테스트임을 명시합니다.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_dev_stack
]

# --- 헬퍼 함수 ---
def load_contract(path: Path) -> dict:
    """YAML 계약 파일을 로드합니다."""
    if not path.is_file():
        pytest.fail(f"계약 파일을 찾을 수 없습니다: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- 테스트 클래스 ---
class TestDevContractCompliance:
    """
    mmp-local-dev 환경이 modern-ml-pipeline이 기대하는 계약을
    준수하는지 검증하는 테스트 스위트.
    """

    @pytest.fixture(scope="class")
    def contracts(self):
        """
        테스트에 필요한 원본 계약과 기대치 계약을 로드하는 픽스처.
        실제 인프라 테스트 전에 계약이 일치하는지 먼저 확인합니다.
        """
        expected = load_contract(EXPECTED_CONTRACT_PATH)
        actual = load_contract(ACTUAL_CONTRACT_PATH)
        return {"expected": expected, "actual": actual}

    def test_contract_consistency(self, contracts):
        """
        [1단계] 원본 계약서와 기대치 계약서의 내용이 일치하는지 검증합니다.
        이 테스트가 실패하면, 인프라 계약에 변경이 발생했다는 의미입니다.
        """
        error_message = (
            "인프라 계약이 변경되었습니다! (../mmp-local-dev/dev-contract.yml)\n"
            "변경 사항을 검토하고 이 파이프라인과 호환되는지 확인 후, "
            "tests/integration/expected-dev-contract.yml 파일을 갱신하세요."
        )
        assert contracts["expected"] == contracts["actual"], error_message

    def test_service_availability(self, contracts):
        """
        [2단계] 계약서에 명시된 모든 서비스가 실제로 실행 중인지 검증합니다.
        이 테스트는 계약 일치 검증이 통과했을 때만 의미가 있습니다.
        """
        services_to_test = contracts["expected"].get("provides_services", [])
        
        if not services_to_test:
            pytest.skip("계약서에 검증할 서비스가 명시되지 않았습니다.")

        for service in services_to_test:
            name = service.get("name")
            port = service.get("port")
            host = "localhost"  # 로컬 개발 환경이므로 localhost로 가정

            try:
                with socket.create_connection((host, port), timeout=5):
                    pass  # 연결 성공
            except (socket.timeout, ConnectionRefusedError) as e:
                pytest.fail(
                    f"계약서에 명시된 서비스 '{name}'에 연결할 수 없습니다 "
                    f"(host: {host}, port: {port}).\n"
                    f"mmp-local-dev 환경이 정상적으로 실행 중인지 확인하세요. 오류: {e}"
                ) 