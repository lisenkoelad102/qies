import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from qies_control.atelier.types import FloatArray, IntArray


def macro_f1(targets: IntArray, predictions: IntArray) -> float:
    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def accuracy(targets: IntArray, predictions: IntArray) -> float:
    return float(accuracy_score(targets, predictions))


def cohen_kappa(targets: IntArray, predictions: IntArray) -> float:
    if np.unique(np.concatenate([targets, predictions])).size < 2:
        return 0.0
    return float(cohen_kappa_score(targets, predictions))


def classification_report(targets: IntArray, predictions: IntArray) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(targets, predictions),
        "accuracy": accuracy(targets, predictions),
        "cohen_kappa": cohen_kappa(targets, predictions),
    }


def as_float_array(values: list[float]) -> FloatArray:
    return np.array(values, dtype=np.float64)
