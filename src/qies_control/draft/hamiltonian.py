from collections import deque
from dataclasses import dataclass

import numpy as np

from qies_control.atelier.types import FloatArray, Molecule
from qies_control.draft.jordan_wigner import annihilation, creation, number
from qies_control.draft.pauli import PauliSum

_ONSITE: dict[str, float] = {
    "C": -0.70,
    "N": -0.90,
    "O": -1.10,
    "S": -0.60,
    "P": -0.55,
    "F": -1.30,
    "Cl": -1.00,
    "Br": -0.95,
    "I": -0.85,
    "H": -0.40,
}
_FAR = 6


def _site_energy(symbol: str) -> float:
    return _ONSITE.get(symbol, -0.75)


def _distance_matrix(molecule: Molecule) -> FloatArray:
    size = max(molecule.n_heavy, 1)
    distance = np.full((size, size), float(_FAR), dtype=np.float64)
    adjacency: list[list[int]] = [[] for _ in range(size)]
    for left, right, _order in molecule.bonds:
        if left < size and right < size:
            adjacency[left].append(right)
            adjacency[right].append(left)
    for source in range(size):
        distance[source, source] = 0.0
        queue: deque[int] = deque([source])
        seen = {source}
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    distance[source, neighbour] = distance[source, node] + 1.0
                    queue.append(neighbour)
    return distance


@dataclass(frozen=True)
class FragmentHamiltonian:
    one_body: FloatArray
    two_body: FloatArray
    n_modes: int

    def to_pauli_sum(self) -> PauliSum:
        operator = PauliSum.zero(self.n_modes)
        for p in range(self.n_modes):
            for q in range(self.n_modes):
                coeff = float(self.one_body[p, q])
                if abs(coeff) < 1e-12:
                    continue
                hop = creation(p, self.n_modes).multiply(annihilation(q, self.n_modes))
                operator = operator.add(hop.scale(coeff))
        for p in range(self.n_modes):
            for q in range(p + 1, self.n_modes):
                weight = float(self.two_body[p, q])
                if abs(weight) < 1e-12:
                    continue
                density = number(p, self.n_modes).multiply(number(q, self.n_modes))
                operator = operator.add(density.scale(weight))
        return operator


def build_fragment(
    left: Molecule, right: Molecule, n_qubits: int, interaction: float = 0.35
) -> FragmentHamiltonian:
    catalogue: list[tuple[str, int, int]] = []
    catalogue.extend((symbol, 0, index) for index, symbol in enumerate(left.symbols))
    catalogue.extend((symbol, 1, index) for index, symbol in enumerate(right.symbols))
    if not catalogue:
        catalogue.append(("C", 0, 0))
    while len(catalogue) < n_qubits:
        catalogue.append(("C", 0, _FAR + len(catalogue)))
    catalogue = catalogue[:n_qubits]

    left_distance = _distance_matrix(left)
    right_distance = _distance_matrix(right)
    one_body = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    two_body = np.zeros((n_qubits, n_qubits), dtype=np.float64)

    for index, (symbol, _origin, _atom) in enumerate(catalogue):
        one_body[index, index] = _site_energy(symbol)

    for p in range(n_qubits):
        for q in range(p + 1, n_qubits):
            _sp, mp, ip = catalogue[p]
            _sq, mq, iq = catalogue[q]
            if mp == mq:
                table = left_distance if mp == 0 else right_distance
                dist = table[ip, iq] if ip < table.shape[0] and iq < table.shape[1] else float(_FAR)
                if dist == 1.0:
                    one_body[p, q] = one_body[q, p] = -0.5
                separation = dist
            else:
                one_body[p, q] = one_body[q, p] = -interaction
                separation = 2.0
            repulsion = 0.6 / (1.0 + separation)
            two_body[p, q] = two_body[q, p] = repulsion

    return FragmentHamiltonian(one_body, two_body, n_qubits)
