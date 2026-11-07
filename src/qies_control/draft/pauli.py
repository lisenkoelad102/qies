from __future__ import annotations

import numpy as np

from qies_control.atelier.types import ComplexArray

PauliString = tuple[int, ...]

_SINGLE: dict[tuple[int, int], tuple[complex, int]] = {
    (0, 0): (1, 0),
    (0, 1): (1, 1),
    (0, 2): (1, 2),
    (0, 3): (1, 3),
    (1, 0): (1, 1),
    (1, 1): (1, 0),
    (1, 2): (1j, 3),
    (1, 3): (-1j, 2),
    (2, 0): (1, 2),
    (2, 1): (-1j, 3),
    (2, 2): (1, 0),
    (2, 3): (1j, 1),
    (3, 0): (1, 3),
    (3, 1): (1j, 2),
    (3, 2): (-1j, 1),
    (3, 3): (1, 0),
}

_DENSE: dict[int, ComplexArray] = {
    0: np.array([[1, 0], [0, 1]], dtype=np.complex128),
    1: np.array([[0, 1], [1, 0]], dtype=np.complex128),
    2: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    3: np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _multiply(left: PauliString, right: PauliString) -> tuple[complex, PauliString]:
    phase: complex = 1.0 + 0.0j
    out = []
    for a, b in zip(left, right, strict=True):
        factor, result = _SINGLE[(a, b)]
        phase *= factor
        out.append(result)
    return phase, tuple(out)


class PauliSum:
    def __init__(self, n_qubits: int, terms: dict[PauliString, complex] | None = None) -> None:
        self.n_qubits = n_qubits
        self.terms: dict[PauliString, complex] = {} if terms is None else dict(terms)

    @classmethod
    def identity(cls, n_qubits: int) -> PauliSum:
        return cls(n_qubits, {(0,) * n_qubits: 1.0 + 0.0j})

    @classmethod
    def zero(cls, n_qubits: int) -> PauliSum:
        return cls(n_qubits, {})

    def add(self, other: PauliSum) -> PauliSum:
        out = dict(self.terms)
        for key, value in other.terms.items():
            out[key] = out.get(key, 0.0 + 0.0j) + value
        return PauliSum(self.n_qubits, out).prune()

    def scale(self, scalar: complex) -> PauliSum:
        return PauliSum(self.n_qubits, {k: scalar * v for k, v in self.terms.items()})

    def multiply(self, other: PauliSum) -> PauliSum:
        out: dict[PauliString, complex] = {}
        for left_key, left_val in self.terms.items():
            for right_key, right_val in other.terms.items():
                phase, key = _multiply(left_key, right_key)
                out[key] = out.get(key, 0.0 + 0.0j) + phase * left_val * right_val
        return PauliSum(self.n_qubits, out).prune()

    def dagger(self) -> PauliSum:
        return PauliSum(self.n_qubits, {k: v.conjugate() for k, v in self.terms.items()})

    def prune(self, tol: float = 1e-12) -> PauliSum:
        self.terms = {k: v for k, v in self.terms.items() if abs(v) > tol}
        return self

    def is_hermitian(self, tol: float = 1e-9) -> bool:
        return all(abs(v.imag) < tol for v in self.terms.values())

    def to_dense(self) -> ComplexArray:
        dim = 2**self.n_qubits
        matrix = np.zeros((dim, dim), dtype=np.complex128)
        for key, value in self.terms.items():
            block: ComplexArray = np.array([[1.0 + 0.0j]], dtype=np.complex128)
            for op in key:
                block = np.asarray(np.kron(block, _DENSE[op]), dtype=np.complex128)
            matrix += value * block
        return matrix
