from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from src.components.fetcher.base import BaseFetcher
from src.utils.core.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings


class FeatureStoreFetcher(BaseFetcher):
    """
    Feature Store(Feast)를 사용하여 피처를 증강하는 fetcher.
    직렬화 가능한 설정만 저장하여 MLflow artifact로 저장/복원 가능.
    """

    def __init__(self, settings: Settings, factory: Any):
        # 직렬화 가능한 설정만 추출
        self._fetcher_config = self._extract_serializable_config(settings)
        self._feature_store_adapter: Optional[Any] = None

        # 학습 시에는 즉시 adapter 생성
        try:
            self._feature_store_adapter = factory.create_feature_store_adapter()
        except Exception as e:
            logger.warning(f"Feature Store adapter 생성 실패 (lazy init 사용): {e}")

    def _extract_serializable_config(self, settings: Settings) -> Dict[str, Any]:
        """직렬화 가능한 설정만 추출"""
        data_interface = settings.recipe.data.data_interface
        fetcher_conf = settings.recipe.data.fetcher
        feast_config = None

        if settings.config.feature_store and settings.config.feature_store.feast_config:
            feast_config = settings.config.feature_store.feast_config.model_dump()

        # feature_views에서 피처 목록 추출
        features: List[str] = []
        feature_views_config = None
        if fetcher_conf and fetcher_conf.feature_views:
            feature_views_config = {}
            for view_name, view_config in fetcher_conf.feature_views.items():
                feature_views_config[view_name] = {
                    "join_key": view_config.join_key,
                    "features": view_config.features,
                }
                for feature in view_config.features:
                    features.append(f"{view_name}:{feature}")

        return {
            "entity_columns": list(data_interface.entity_columns),
            "timestamp_column": fetcher_conf.timestamp_column if fetcher_conf else None,
            "target_column": data_interface.target_column,
            "treatment_column": getattr(data_interface, "treatment_column", None),
            "feature_views": feature_views_config,
            "features": features,
            "feast_config": feast_config,
        }

    def _get_adapter(self):
        """Lazy initialization으로 adapter 획득"""
        if self._feature_store_adapter is None:
            logger.info("Feature Store adapter를 lazy initialization으로 생성합니다.")
            try:
                from src.components.adapter.modules.feast_adapter import FeastAdapter
                from src.settings.config import FeastConfig

                feast_config_dict = self._fetcher_config.get("feast_config")
                if not feast_config_dict:
                    raise ValueError("Feast 설정이 없습니다. artifact가 손상되었을 수 있습니다.")

                # 간이 Settings 객체 생성 (FeastAdapter 호환용)
                class MinimalSettings:
                    class Config:
                        class FeatureStore:
                            feast_config = FeastConfig(**feast_config_dict)

                        feature_store = FeatureStore()

                    config = Config()

                self._feature_store_adapter = FeastAdapter(MinimalSettings())
            except Exception as e:
                logger.error(f"Feature Store adapter lazy init 실패: {e}")
                raise

        return self._feature_store_adapter

    def __getstate__(self):
        """Pickle 직렬화 시 adapter 제외"""
        state = self.__dict__.copy()
        state["_feature_store_adapter"] = None
        return state

    def __setstate__(self, state):
        """Pickle 역직렬화"""
        self.__dict__.update(state)

    def fetch(self, df: pd.DataFrame, run_mode: str = "batch") -> pd.DataFrame:
        """
        피처 증강 수행.

        Args:
            df: 입력 DataFrame
            run_mode: "train", "batch", "serving" 중 하나

        Returns:
            증강된 DataFrame
        """
        logger.info(f"Feature Store 피처 증강 시작 (run_mode={run_mode})")

        adapter = self._get_adapter()
        entity_columns = self._fetcher_config["entity_columns"]
        features = self._fetcher_config["features"]

        data_interface_config = {
            "entity_columns": entity_columns,
            "timestamp_column": self._fetcher_config["timestamp_column"],
            "target_column": self._fetcher_config["target_column"],
            "treatment_column": self._fetcher_config["treatment_column"],
        }

        if run_mode in ("train", "batch"):
            # 오프라인 스토어: Point-in-Time 검증 포함
            result = adapter.get_historical_features_with_validation(
                entity_df=df,
                features=features,
                data_interface_config=data_interface_config,
            )
            logger.info(f"피처 증강 완료 (offline): {result.shape}")
            return result

        elif run_mode == "serving":
            # 온라인 스토어: 클라이언트 제공 피처 보존 + Online Store 피처 병합
            client_df = df.copy()

            # entity만 추출하여 Online Store 조회
            entity_rows = df[entity_columns].to_dict(orient="records")
            online_features_df = adapter.get_online_features(
                entity_rows=entity_rows,
                features=features,
            )

            # 혼용 모드: 클라이언트 제공 피처 + Online Store 피처 병합
            # Online Store 결과에서 entity 컬럼 제외 (중복 방지)
            online_feature_cols = [
                col for col in online_features_df.columns if col not in entity_columns
            ]

            if online_feature_cols:
                # entity 기준으로 병합
                result = client_df.merge(
                    online_features_df[entity_columns + online_feature_cols],
                    on=entity_columns,
                    how="left",
                )
            else:
                result = client_df

            logger.info(f"피처 증강 완료 (online, 혼용): {result.shape}")
            return result

        else:
            raise ValueError(f"지원하지 않는 run_mode: {run_mode}")


# Self-registration
from ..registry import FetcherRegistry

FetcherRegistry.register("feature_store", FeatureStoreFetcher)
