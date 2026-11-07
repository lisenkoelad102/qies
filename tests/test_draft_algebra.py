import numpy as np

from qies_control.atelier.types import Molecule
from qies_control.draft.hamiltonian import build_fragment
from qies_control.draft.jordan_wigner import annihilation, creation, number
from qies_control.draft.pauli import PauliSum


def test_number_operator_matches_dagger_product() -> None:
    n_modes = 3
    for mode in range(n_modes):
        direct = number(mode, n_modes).to_dense()
        composed = creation(mode, n_modes).multiply(annihilation(mode, n_modes)).to_dense()
        assert np.allclose(direct, composed)


def test_fragment_hamiltonian_is_hermitian() -> None:
    left = Molecule(("C", "N", "O"), ((0, 1, 1.0), (1, 2, 2.0)))
    right = Molecule(("C", "C"), ((0, 1, 1.0),))
    operator = build_fragment(left, right, 4).to_pauli_sum()
    assert operator.is_hermitian()
    dense = operator.to_dense()
    assert np.allclose(dense, dense.conj().T)


def test_pauli_multiplication_phase() -> None:
    x = PauliSum(1, {(1,): 1.0 + 0.0j})
    y = PauliSum(1, {(2,): 1.0 + 0.0j})
    product = x.multiply(y)
    assert product.terms == {(3,): 1j}
