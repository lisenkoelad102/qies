import numpy as np

from qies_control.atelier.settings import VqeConfig
from qies_control.atelier.types import FloatArray
from qies_control.canvas.ansatz import hardware_efficient, parameter_count
from qies_control.canvas.spsa import spsa_minimize
from qies_control.canvas.statevector import expectation
from qies_control.draft.pauli import PauliSum


def estimate_energy(operator: PauliSum, config: VqeConfig) -> float:
    n_qubits = operator.n_qubits
    depth = config.depth
    rng = np.random.default_rng(config.seed)
    x0: FloatArray = rng.normal(0.0, 0.1, size=parameter_count(n_qubits, depth))

    def objective(params: FloatArray) -> float:
        state = hardware_efficient(params, n_qubits, depth)
        return expectation(state, operator)

    _, value = spsa_minimize(
        objective,
        x0,
        config.spsa_iters,
        config.spsa_a,
        config.spsa_c,
        config.spsa_alpha,
        config.spsa_gamma,
        config.seed,
    )
    return value
