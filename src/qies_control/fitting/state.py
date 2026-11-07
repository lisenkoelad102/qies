import numpy as np

from qies_control.atelier.types import FloatArray


def augment_state(state: FloatArray, potential: float) -> FloatArray:
    return np.concatenate([state, np.array([potential], dtype=np.float64)])


def burden_signal(state: FloatArray, patient_phi: float, gain: float = 4.0) -> float:
    drug_level = float(state[0])
    return patient_phi * float(1.0 / (1.0 + np.exp(-gain * (drug_level - 0.5))))
