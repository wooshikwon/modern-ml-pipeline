from ._augmenter import FeatureStoreAugmenter
from ._pass_through import PassThroughAugmenter

# 선택적으로 SQL Fallback Augmenter가 추가되면 아래에서 임포트/등록
# from ._sql_fallback import SqlFallbackAugmenter  # to be implemented

try:
    from src.engine._registry import AugmenterRegistry
    AugmenterRegistry.register("feature_store", FeatureStoreAugmenter)
    AugmenterRegistry.register("pass_through", PassThroughAugmenter)
    # AugmenterRegistry.register("sql_fallback", SqlFallbackAugmenter)
except Exception:
    # 레지스트리 초기 임포트 순서에 따라 무시 가능
    pass 