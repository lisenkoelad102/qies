import numpy as np
from scipy import stats

from qies_control.atelier.types import FloatArray


def bootstrap_ci(
    values: FloatArray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = np.array(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(n_boot)]
    )
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(array.mean()), float(low), float(high)


def paired_wilcoxon(left: FloatArray, right: FloatArray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if np.allclose(difference, 0.0):
        return 1.0
    result = stats.wilcoxon(difference)
    return float(result.pvalue)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    array = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty(array.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (array.size - rank) * array[index]
        running = max(running, scaled)
        adjusted[index] = min(running, 1.0)
    return [float(value) for value in adjusted]


def tost_equivalence(left: FloatArray, right: FloatArray, margin: float) -> tuple[bool, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    sample_size = difference.size
    standard_error = difference.std(ddof=1) / np.sqrt(sample_size)
    if standard_error <= 0.0:
        equivalent = abs(float(difference.mean())) < margin
        return equivalent, 0.0 if equivalent else 1.0
    degrees = sample_size - 1
    t_lower = (difference.mean() + margin) / standard_error
    t_upper = (difference.mean() - margin) / standard_error
    p_lower = float(1.0 - stats.t.cdf(t_lower, degrees))
    p_upper = float(stats.t.cdf(t_upper, degrees))
    p_value = max(p_lower, p_upper)
    return bool(p_value < 0.05), p_value
