from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, List, Dict, Any
from src.interface import BaseFetcher
from src.utils.system.console_manager import get_console

if TYPE_CHECKING:
    from src.settings import Settings


class FeatureStoreFetcher(BaseFetcher):
    """
    Feature Store(Feast)를 사용하여 피처를 증강하는 fetcher.
    DEV/PROD 환경에서 사용됩니다.
    """
    def __init__(self, settings: Settings, factory: Any):
        self.settings = settings
        self.factory = factory
        self.feature_store_adapter = self.factory.create_feature_store_adapter()

    def fetch(self, df: pd.DataFrame, run_mode: str = "batch") -> pd.DataFrame:
        console = get_console()
        console.info("Feature Store를 통해 피처 증강을 시작합니다.",
                    rich_message="🏦 Starting feature augmentation via Feature Store")

        # ✅ 새로운 구조에서 설정 수집
        data_interface = self.settings.recipe.data.data_interface
        fetcher_conf = self.settings.recipe.data.fetcher

        # ✅ 새로운 feature_views 구조에서 features 리스트 구성
        features: List[str] = []
        if fetcher_conf and fetcher_conf.feature_views:
            for view_name, view_config in fetcher_conf.feature_views.items():
                for feature in view_config.features:
                    features.append(f"{view_name}:{feature}")

        # ✅ 새로운 구조로 data_interface_config 구성
        data_interface_config: Dict[str, Any] = {
            'entity_columns': data_interface.entity_columns,
            'timestamp_column': fetcher_conf.timestamp_column if fetcher_conf else None,
            'task_type': data_interface.task_type,
            'target_column': data_interface.target_column,
            'treatment_column': getattr(data_interface, 'treatment_column', None),
        }

        if run_mode in ("train", "batch"):
            # 오프라인 PIT 조회 + 검증
            result = self.feature_store_adapter.get_historical_features_with_validation(
                entity_df=df,
                features=features,
                data_interface_config=data_interface_config,
            )
            console.info("피처 증강 완료(offline).",
                        rich_message="✅ Feature augmentation complete (offline)")
            return result
        elif run_mode == "serving":
            # 온라인 조회: entity_rows(dict list)로 변환 필요
            entity_rows = df[data_interface.entity_columns].to_dict(orient="records")
            result = self.feature_store_adapter.get_online_features(
                entity_rows=entity_rows,
                features=features,
            )
            console.info("피처 증강 완료(online).",
                        rich_message="✅ Feature augmentation complete (online)")
            return result
        else:
            raise ValueError(f"Unsupported run_mode: {run_mode}")

# Self-registration
from ..registry import FetcherRegistry
FetcherRegistry.register("feature_store", FeatureStoreFetcher)
