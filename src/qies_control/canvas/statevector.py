import numpy as np

from qies_control.atelier.types import ComplexArray, FloatArray
from qies_control.draft.pauli import PauliSum

_PAULI_DENSE: dict[int, ComplexArray] = {
    1: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    2: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    3: np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def zero_state(n_qubits: int) -> ComplexArray:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def plus_state(n_qubits: int) -> ComplexArray:
    amplitude = 1.0 / np.sqrt(2.0**n_qubits)
    return np.full(2**n_qubits, amplitude, dtype=np.complex128)


def _apply_single(
    state: ComplexArray, gate: ComplexArray, qubit: int, n_qubits: int
) -> ComplexArray:
    tensor = state.reshape([2] * n_qubits)
    contracted = np.tensordot(gate, tensor, axes=([1], [qubit]))
    moved = np.moveaxis(contracted, 0, qubit)
    return np.asarray(moved.reshape(-1), dtype=np.complex128)


def apply_ry(state: ComplexArray, angle: float, qubit: int, n_qubits: int) -> ComplexArray:
    half = angle / 2.0
    gate = np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]], dtype=np.complex128
    )
    return _apply_single(state, gate, qubit, n_qubits)


def apply_rz(state: ComplexArray, angle: float, qubit: int, n_qubits: int) -> ComplexArray:
    half = angle / 2.0
    gate = np.array([[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]], dtype=np.complex128)
    return _apply_single(state, gate, qubit, n_qubits)


def apply_rx(state: ComplexArray, angle: float, qubit: int, n_qubits: int) -> ComplexArray:
    half = angle / 2.0
    gate = np.array(
        [[np.cos(half), -1j * np.sin(half)], [-1j * np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )
    return _apply_single(state, gate, qubit, n_qubits)


def apply_cnot(state: ComplexArray, control: int, target: int, n_qubits: int) -> ComplexArray:
    index = np.arange(2**n_qubits)
    control_bit = (index >> (n_qubits - 1 - control)) & 1
    permutation = index ^ (control_bit << (n_qubits - 1 - target))
    return np.asarray(state[permutation], dtype=np.complex128)


def _apply_pauli_string(state: ComplexArray, key: tuple[int, ...], n_qubits: int) -> ComplexArray:
    out = state
    for qubit, op in enumerate(key):
        if op != 0:
            out = _apply_single(out, _PAULI_DENSE[op], qubit, n_qubits)
    return out


def expectation(state: ComplexArray, operator: PauliSum) -> float:
    total = 0.0 + 0.0j
    for key, coeff in operator.terms.items():
        applied = _apply_pauli_string(state, key, operator.n_qubits)
        total += coeff * complex(np.vdot(state, applied))
    return float(total.real)


def diagonal_expectation(amplitudes: ComplexArray, diagonal: FloatArray) -> float:
    probabilities = np.abs(amplitudes) ** 2
    return float(np.dot(probabilities, diagonal))
