import numpy as np

from qies_control.atelier.settings import QaoaConfig, VqeConfig
from qies_control.atelier.types import Molecule
from qies_control.canvas.qaoa import burden_potential, exact_max_cut
from qies_control.canvas.spsa import spsa_minimize
from qies_control.canvas.vqe import estimate_energy
from qies_control.draft.hamiltonian import build_fragment


def test_vqe_respects_variational_bound() -> None:
    left = Molecule(("C", "C", "O"), ((0, 1, 1.0), (1, 2, 1.0)))
    right = Molecule(("N", "C"), ((0, 1, 1.0),))
    operator = build_fragment(left, right, 4).to_pauli_sum()
    ground_state = float(np.linalg.eigvalsh(operator.to_dense()).min())
    estimate = estimate_energy(operator, VqeConfig(n_qubits=4, depth=6, spsa_iters=80, seed=3))
    assert estimate >= ground_state - 1e-6


def test_qaoa_recovers_max_cut() -> None:
    coupling = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (3, 0): 1.0, (0, 2): 1.0}
    exact = exact_max_cut(coupling, 4)
    phi = burden_potential(coupling, 4, QaoaConfig(p_layers=2, opt_iters=150, seed=5))
    assert phi <= exact + 1e-6
    assert phi >= 0.6 * exact


def test_spsa_minimises_quadratic() -> None:
    target = np.array([1.5, -2.0])

    def objective(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    start = np.zeros(2)
    solution, value = spsa_minimize(objective, start, n_iter=200, a=0.2, c=0.1, seed=0)
    assert value < objective(start)
    assert np.linalg.norm(solution - target) < 1.0
