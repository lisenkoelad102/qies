import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from qies_control.atelier.types import FloatArray, IntArray


def _valid_columns(targets: IntArray) -> IntArray:
    keep = [
        index for index in range(targets.shape[1]) if 0 < targets[:, index].sum() < targets.shape[0]
    ]
    return np.array(keep, dtype=np.int64)


def auroc(targets: IntArray, scores: FloatArray) -> float:
    columns = _valid_columns(targets)
    if columns.size == 0:
        return 0.5
    return float(roc_auc_score(targets[:, columns], scores[:, columns], average="macro"))


def auprc(targets: IntArray, scores: FloatArray) -> float:
    columns = _valid_columns(targets)
    if columns.size == 0:
        return 0.0
    return float(average_precision_score(targets[:, columns], scores[:, columns], average="macro"))


def ranking_report(targets: IntArray, scores: FloatArray) -> dict[str, float]:
    return {"auroc": auroc(targets, scores), "auprc": auprc(targets, scores)}
