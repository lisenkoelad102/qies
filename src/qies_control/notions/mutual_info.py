import numpy as np

from qies_control.atelier.types import FloatArray, IntArray


def _entropy(counts: FloatArray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def discrete_mutual_information(labels: IntArray, codes: IntArray) -> float:
    joint = np.zeros((labels.max() + 1, codes.max() + 1), dtype=np.float64)
    for label, code in zip(labels, codes, strict=True):
        joint[label, code] += 1.0
    marginal_label = joint.sum(axis=1)
    marginal_code = joint.sum(axis=0)
    return _entropy(marginal_label) + _entropy(marginal_code) - _entropy(joint.reshape(-1))


def quantize(values: FloatArray, n_bins: int) -> IntArray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(values, edges).astype(np.int64)
