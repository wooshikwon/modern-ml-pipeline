"""
AWS S3 service checker implementation
Phase 6: Universal system-check architecture

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, Optional, List
import re

from ..base import BaseServiceChecker
from ..models import CheckResult


class S3Checker(BaseServiceChecker):
    """
    AWS S3 스토리지 연결 검사 체커.

    Config에서 s3:// URI가 포함된 설정을 찾아 S3 접근성을 검증합니다.

    검사 대상 설정:
    - artifacts.model_registry.storage_uri
    - data_adapters에서 s3:// 경로들
    - 기타 s3:// 프로토콜을 사용하는 모든 설정
    """

    def can_check(self, config: Dict[str, Any]) -> bool:
        """
        Config에 s3:// URI가 포함되어 있는지 확인.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: s3:// URI가 설정되어 있으면 True
        """
        s3_uris = self._find_s3_uris(config)
        return len(s3_uris) > 0

    def check(self, config: Dict[str, Any]) -> CheckResult:
        """
        AWS S3 연결 검사 수행.

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            CheckResult: S3 연결 검사 결과
        """
        s3_uris = self._find_s3_uris(config)

        if not s3_uris:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="S3 URI 설정이 없음",
                recommendations=["config에 s3:// 형식의 URI를 설정하세요"],
            )

        try:
            # boto3를 사용한 실제 S3 연결 테스트
            return self._test_s3_connection(s3_uris)

        except ImportError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="AWS SDK(boto3)가 설치되지 않음",
                recommendations=["boto3 설치: pip install boto3", "또는 uv add boto3"],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"S3 연결 검사 중 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def get_service_name(self) -> str:
        """서비스 이름 반환."""
        return "AWS S3"

    def _find_s3_uris(self, config: Dict[str, Any]) -> List[str]:
        """Config에서 모든 s3:// URI를 재귀적으로 찾기."""
        s3_uris = []

        def _recursive_search(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    _recursive_search(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _recursive_search(item, f"{path}[{i}]")
            elif isinstance(obj, str) and obj.startswith("s3://"):
                s3_uris.append(obj)

        _recursive_search(config)
        return s3_uris

    def _test_s3_connection(self, s3_uris: List[str]) -> CheckResult:
        """실제 S3 연결 테스트."""
        try:
            import boto3
            from botocore.exceptions import (
                NoCredentialsError,
                BotoCoreError,
                ClientError,
            )

            # S3 클라이언트 생성
            s3_client = boto3.client("s3")

            # 첫 번째 URI를 사용해서 버킷 접근 테스트
            test_uri = s3_uris[0]
            bucket_name = self._extract_bucket_name(test_uri)

            if not bucket_name:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"잘못된 S3 URI 형식: {test_uri}",
                    recommendations=["올바른 s3://bucket-name/path 형식을 사용하세요"],
                )

            # head_bucket으로 버킷 존재 여부 및 접근 권한 확인
            s3_client.head_bucket(Bucket=bucket_name)

            # 성공한 경우 버킷 정보 수집
            try:
                location = s3_client.get_bucket_location(Bucket=bucket_name)
                region = location.get("LocationConstraint") or "us-east-1"

                # 간단한 list_objects 테스트로 읽기 권한 확인
                response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                object_count = response.get("KeyCount", 0)

                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"S3 버킷 접근 성공: {bucket_name}",
                    details=[
                        f"버킷 리전: {region}",
                        f"오브젝트 수 (샘플): {object_count}개",
                        f"총 S3 URI 수: {len(s3_uris)}",
                        f"테스트 URI: {test_uri}",
                    ],
                )

            except ClientError as inner_e:
                # head_bucket은 성공했지만 세부 정보 조회 실패
                return CheckResult(
                    is_healthy=True,
                    service_name=self.get_service_name(),
                    message=f"S3 버킷 기본 접근 성공: {bucket_name}",
                    details=[
                        f"세부 정보 조회 제한: {inner_e.response['Error']['Code']}",
                        f"총 S3 URI 수: {len(s3_uris)}",
                    ],
                )

        except NoCredentialsError:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message="AWS 자격 증명을 찾을 수 없음",
                recommendations=[
                    "AWS 자격 증명을 설정하세요: aws configure",
                    "또는 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 환경변수 설정",
                    "또는 IAM 역할을 사용하세요 (EC2/Lambda에서)",
                ],
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "404" or "NoSuchBucket" in error_code:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"S3 버킷을 찾을 수 없음: {bucket_name}",
                    recommendations=[
                        f"버킷이 존재하는지 확인하세요: {bucket_name}",
                        "AWS Console에서 버킷을 확인하세요",
                        "버킷이 다른 리전에 있는지 확인하세요",
                    ],
                )
            elif error_code == "403" or "AccessDenied" in error_code:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"S3 버킷 접근 권한 없음: {bucket_name}",
                    recommendations=[
                        "AWS 자격 증명의 S3 권한을 확인하세요",
                        f"버킷 {bucket_name}에 대한 s3:ListBucket 권한이 필요합니다",
                        "IAM 정책을 확인하세요",
                    ],
                )
            else:
                return CheckResult(
                    is_healthy=False,
                    service_name=self.get_service_name(),
                    message=f"S3 접근 오류 ({error_code}): {error_message}",
                    recommendations=self._generate_error_recommendations(error_message),
                )
        except BotoCoreError as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"AWS SDK 설정 오류: {str(e)}",
                recommendations=[
                    "AWS 설정을 확인하세요: aws configure list",
                    "AWS 리전이 올바르게 설정되어 있는지 확인하세요",
                ],
            )
        except Exception as e:
            return CheckResult(
                is_healthy=False,
                service_name=self.get_service_name(),
                message=f"S3 연결 테스트 중 예상치 못한 오류: {str(e)}",
                recommendations=self._generate_error_recommendations(str(e)),
            )

    def _extract_bucket_name(self, s3_uri: str) -> Optional[str]:
        """s3:// URI에서 버킷 이름 추출."""
        # s3://bucket-name/path/to/file => bucket-name
        match = re.match(r"s3://([^/]+)", s3_uri)
        return match.group(1) if match else None

    def _generate_error_recommendations(self, error_message: str) -> List[str]:
        """에러 메시지를 기반으로 해결 권장사항 생성."""
        recommendations = [
            "S3 URI 형식을 확인하세요: s3://bucket-name/path",
            "AWS 자격 증명과 권한을 확인하세요",
        ]

        if "timeout" in error_message.lower():
            recommendations.extend(
                ["네트워크 연결 상태를 확인하세요", "AWS S3 서비스 상태를 확인하세요"]
            )
        elif "credential" in error_message.lower():
            recommendations.extend(
                [
                    "AWS 자격 증명을 확인하세요: aws configure",
                    "IAM 사용자의 S3 권한을 확인하세요",
                ]
            )
        elif "region" in error_message.lower():
            recommendations.append("AWS 리전 설정을 확인하세요")

        return recommendations
