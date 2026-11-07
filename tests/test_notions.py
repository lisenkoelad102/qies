import numpy as np

from qies_control.notions.mutual_info import discrete_mutual_information, quantize
from qies_control.notions.stats import (
    bootstrap_ci,
    holm_bonferroni,
    paired_wilcoxon,
    tost_equivalence,
)


def test_information_advantage_inequality() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=4000)
    energy = labels.astype(np.float64) + rng.normal(0.0, 0.3, size=labels.size)
    codes = quantize(energy, n_bins=8)
    quantized_label = quantize(energy, n_bins=2)
    mi_energy = discrete_mutual_information(labels, codes)
    mi_label = discrete_mutual_information(labels, quantized_label)
    assert mi_energy >= mi_label - 1e-9


def test_stats_sanity() -> None:
    rng = np.random.default_rng(1)
    left = rng.normal(1.0, 0.1, size=40)
    right = left + rng.normal(0.0, 0.01, size=40)
    mean, low, high = bootstrap_ci(left, n_boot=400, seed=2)
    assert low <= mean <= high
    equivalent, p_value = tost_equivalence(left, right, margin=0.2)
    assert equivalent
    assert 0.0 <= paired_wilcoxon(left, right) <= 1.0
    adjusted = holm_bonferroni([0.01, 0.04, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
