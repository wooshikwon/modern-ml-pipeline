"""Monitor 공통 유틸리티."""

import numpy as np

NUM_BINS = 10
EPSILON = 1e-4


def compute_psi(
    values: np.ndarray,
    bin_edges: list[float],
    ref_proportions: list[float],
    epsilon: float = EPSILON,
) -> float:
    """Population Stability Index를 계산한다."""
    counts = np.histogram(values, bins=bin_edges)[0]
    total = counts.sum()
    if total == 0:
        return 0.0
    actual = counts / total
    ref = np.array(ref_proportions)
    actual = np.clip(actual, epsilon, None)
    ref = np.clip(ref, epsilon, None)
    return float(np.sum((actual - ref) * np.log(actual / ref)))
