import numpy as np

from qies_control.atelier.types import FloatArray, IntArray


def time_in_therapeutic_range(in_range: IntArray) -> float:
    if in_range.size == 0:
        return 0.0
    return float(np.mean(in_range)) * 100.0


def adverse_interaction_exposure(adverse: FloatArray) -> float:
    if adverse.size == 0:
        return 0.0
    return float(np.mean(adverse)) * 100.0


def mucosal_healing_proxy(healing: FloatArray) -> float:
    if healing.size == 0:
        return 0.0
    return float(np.mean(healing)) * 100.0
