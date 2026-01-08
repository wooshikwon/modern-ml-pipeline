"""Config-Recipe 간 호환성 충돌 검증"""

from ..config import Config
from ..recipe import Recipe
from .common import ValidationResult


class CompatibilityValidator:
    """Config와 Recipe 간 설정 충돌 검증"""

    def validate_feature_store_consistency(
        self, config: Config, recipe: Recipe
    ) -> ValidationResult:
        """Feature Store 설정 일관성 검증"""
        recipe_fetcher_type = recipe.data.fetcher.type
        config_fs_provider = config.feature_store.provider

        if recipe_fetcher_type == "feature_store":
            # Recipe에서 feature_store 사용 시 Config에 설정 필요
            if config_fs_provider == "none":
                return ValidationResult(
                    is_valid=False,
                    error_message="Recipe에서 feature_store fetcher를 사용하지만 Config feature_store provider가 'none'입니다.",
                )

            # Feast 설정 존재 확인
            if config_fs_provider == "feast" and not config.feature_store.feast_config:
                return ValidationResult(
                    is_valid=False,
                    error_message="Recipe에서 feature_store를 사용하지만 Config에 feast_config가 없습니다.",
                )

        elif config_fs_provider != "none":
            # Config에 Feature Store 설정했지만 Recipe에서 사용 안함 - 경고만
            return ValidationResult(
                is_valid=True,
                warnings=[
                    "Config에 feature_store 설정이 있지만 Recipe에서 feature_store fetcher를 사용하지 않습니다."
                ],
            )

        return ValidationResult(is_valid=True)

    def validate_data_source_compatibility(
        self, config: Config, recipe: Recipe
    ) -> ValidationResult:
        """데이터 소스 어댑터 호환성 검증"""
        # source_uri가 주입된 이후에만 검증 가능
        if not recipe.data.loader.source_uri:
            return ValidationResult(is_valid=True)

        source_uri = recipe.data.loader.source_uri.lower()
        config_adapter = config.data_source.adapter_type

        # URI 패턴 기반 어댑터 타입 추론
        if self._is_sql_pattern(source_uri):
            compatible_types = ["sql", "bigquery"]
            if config_adapter not in compatible_types:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"SQL 데이터 소스는 adapter_type이 {compatible_types} 중 하나여야 합니다. 현재: {config_adapter}",
                )

        elif self._is_storage_pattern(source_uri):
            if config_adapter != "storage":
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Storage 파일은 adapter_type이 'storage'여야 합니다. 현재: {config_adapter}",
                )

        elif source_uri.startswith("bigquery://"):
            if config_adapter != "sql":  # BigQuery는 sql adapter 사용
                return ValidationResult(
                    is_valid=False,
                    error_message=f"BigQuery URI는 adapter_type이 'sql'이어야 합니다. 현재: {config_adapter}",
                )

        return ValidationResult(is_valid=True)

    def validate_mlflow_settings(self, config: Config) -> ValidationResult:
        """MLflow 설정 검증 (경고만)"""
        if not config.mlflow:
            return ValidationResult(
                is_valid=True,
                warnings=["MLflow가 설정되지 않았습니다. 실험 추적이 비활성화됩니다."],
            )
        return ValidationResult(is_valid=True)

    def _is_sql_pattern(self, uri: str) -> bool:
        """SQL 패턴 검사"""
        return uri.endswith(".sql") or "select" in uri or "from" in uri

    def _is_storage_pattern(self, uri: str) -> bool:
        """Storage 패턴 검사"""
        return uri.endswith((".csv", ".parquet", ".json")) or uri.startswith(
            ("s3://", "gs://", "az://")
        )
