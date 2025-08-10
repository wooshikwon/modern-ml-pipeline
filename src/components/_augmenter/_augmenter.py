from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, List, Dict, Any
from src.interface import BaseAugmenter
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings
    from src.engine import Factory


class FeatureStoreAugmenter(BaseAugmenter):
    """
    Feature Store(Feast)를 사용하여 피처를 증강하는 Augmenter.
    DEV/PROD 환경에서 사용됩니다.
    """
    def __init__(self, settings: Settings, factory: Factory):
        self.settings = settings
        self.factory = factory
        self.feature_store_adapter = self.factory.create_feature_store_adapter()

    def augment(self, df: pd.DataFrame, run_mode: str = "batch") -> pd.DataFrame:
        logger.info("Feature Store를 통해 피처 증강을 시작합니다.")

        model_cfg = self.settings.recipe.model
        entity_schema = model_cfg.loader.entity_schema
        augmenter_cfg = model_cfg.augmenter
        data_interface = model_cfg.data_interface

        # features 리스트 구성 (namespace별 features -> 평탄화 문자열 리스트)
        features: List[str] = []
        if augmenter_cfg and augmenter_cfg.features:
            for ns in augmenter_cfg.features:
                for f in ns.features:
                    features.append(f"{ns.feature_namespace}:{f}")

        data_interface_config: Dict[str, Any] = {
            'entity_columns': entity_schema.entity_columns,
            'timestamp_column': entity_schema.timestamp_column,
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
            logger.info("피처 증강 완료(offline).")
            return result
        elif run_mode == "serving":
            # 온라인 조회: entity_rows(dict list)로 변환 필요
            entity_rows = df[entity_schema.entity_columns].to_dict(orient="records")
            result = self.feature_store_adapter.get_online_features(
                entity_rows=entity_rows,
                features=features,
            )
            logger.info("피처 증강 완료(online).")
            return result
        else:
            raise ValueError(f"Unsupported run_mode: {run_mode}")
