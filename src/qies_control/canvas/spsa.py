from collections.abc import Callable

import numpy as np

from qies_control.atelier.types import FloatArray


def spsa_minimize(
    objective: Callable[[FloatArray], float],
    x0: FloatArray,
    n_iter: int,
    a: float,
    c: float,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: int = 0,
) -> tuple[FloatArray, float]:
    rng = np.random.default_rng(seed)
    x = x0.astype(np.float64).copy()
    stability = max(1, n_iter // 10)
    best_x = x.copy()
    best_value = objective(x)
    for k in range(n_iter):
        ak = a / ((k + 1 + stability) ** alpha)
        ck = c / ((k + 1) ** gamma)
        delta = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
        plus = objective(x + ck * delta)
        minus = objective(x - ck * delta)
        gradient = (plus - minus) / (2.0 * ck) * delta
        x = x - ak * gradient
        value = objective(x)
        if value < best_value:
            best_value = value
            best_x = x.copy()
    return best_x, best_value
