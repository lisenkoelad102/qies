from qies_control.atelier.settings import VqeConfig
from qies_control.canvas.vqe import estimate_energy
from qies_control.cloth.graph import coupling_from_energy
from qies_control.cloth.smiles import parse_smiles
from qies_control.draft.hamiltonian import build_fragment


def interaction_energy(left_smiles: str, right_smiles: str, config: VqeConfig) -> dict[str, float]:
    left = parse_smiles(left_smiles)
    right = parse_smiles(right_smiles)
    fragment = build_fragment(left, right, config.n_qubits)
    energy = estimate_energy(fragment.to_pauli_sum(), config)
    return {
        "energy": energy,
        "coupling": coupling_from_energy(energy),
        "left_heavy_atoms": float(left.n_heavy),
        "right_heavy_atoms": float(right.n_heavy),
    }
