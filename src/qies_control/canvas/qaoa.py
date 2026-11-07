from collections.abc import Mapping

import numpy as np

from qies_control.atelier.settings import QaoaConfig
from qies_control.atelier.types import FloatArray
from qies_control.canvas.spsa import spsa_minimize
from qies_control.canvas.statevector import apply_rx, diagonal_expectation, plus_state

Coupling = Mapping[tuple[int, int], float]


def cost_diagonal(coupling: Coupling, n_nodes: int) -> FloatArray:
    dim = 2**n_nodes
    index = np.arange(dim)
    diagonal = np.zeros(dim, dtype=np.float64)
    for (left, right), weight in coupling.items():
        spin_left = 1 - 2 * ((index >> (n_nodes - 1 - left)) & 1)
        spin_right = 1 - 2 * ((index >> (n_nodes - 1 - right)) & 1)
        diagonal += weight * 0.5 * (1 - spin_left * spin_right)
    return diagonal


def exact_max_cut(coupling: Coupling, n_nodes: int) -> float:
    if n_nodes == 0:
        return 0.0
    return float(cost_diagonal(coupling, n_nodes).max())


def burden_potential(coupling: Coupling, n_nodes: int, config: QaoaConfig) -> float:
    if n_nodes < 2 or not coupling:
        return 0.0
    diagonal = cost_diagonal(coupling, n_nodes)
    layers = config.p_layers

    def negative_value(params: FloatArray) -> float:
        gammas = params[:layers]
        betas = params[layers:]
        amplitudes = plus_state(n_nodes)
        for layer in range(layers):
            amplitudes = amplitudes * np.exp(-1j * gammas[layer] * diagonal)
            for qubit in range(n_nodes):
                amplitudes = apply_rx(amplitudes, 2.0 * betas[layer], qubit, n_nodes)
        return -diagonal_expectation(amplitudes, diagonal)

    rng = np.random.default_rng(config.seed)
    x0: FloatArray = rng.uniform(0.0, np.pi, size=2 * layers)
    _, value = spsa_minimize(
        negative_value, x0, config.opt_iters, config.step, config.step, seed=config.seed
    )
    return -value
