from dataclasses import dataclass

import numpy as np

from qies_control.atelier.settings import VqeConfig
from qies_control.atelier.types import FloatArray, InteractionGraph, Molecule
from qies_control.canvas.vqe import estimate_energy
from qies_control.draft.hamiltonian import build_fragment

_HALOGENS = {"F", "Cl", "Br", "I"}


@dataclass(frozen=True)
class Patient:
    agents: tuple[Molecule, ...]
    pairs: tuple[tuple[int, int], ...]
    relations: tuple[int, ...]
    nutrient_flags: tuple[bool, ...]


def coupling_from_energy(energy: float, scale: float = 1.0) -> float:
    return scale * abs(energy)


def node_feature(molecule: Molecule, feature_dim: int) -> FloatArray:
    counts = {
        "heavy": float(molecule.n_heavy),
        "carbon": float(sum(s == "C" for s in molecule.symbols)),
        "nitrogen": float(sum(s == "N" for s in molecule.symbols)),
        "oxygen": float(sum(s == "O" for s in molecule.symbols)),
        "sulfur": float(sum(s == "S" for s in molecule.symbols)),
        "halogen": float(sum(s in _HALOGENS for s in molecule.symbols)),
        "bonds": float(len(molecule.bonds)),
        "order": float(np.mean([b[2] for b in molecule.bonds]) if molecule.bonds else 0.0),
    }
    vector = np.array(list(counts.values()), dtype=np.float64)
    if vector.shape[0] >= feature_dim:
        return vector[:feature_dim]
    padded = np.zeros(feature_dim, dtype=np.float64)
    padded[: vector.shape[0]] = vector
    return padded


def precompute_energies(patient: Patient, config: VqeConfig) -> dict[tuple[int, int], float]:
    energies: dict[tuple[int, int], float] = {}
    for left, right in patient.pairs:
        fragment = build_fragment(patient.agents[left], patient.agents[right], config.n_qubits)
        energies[(left, right)] = estimate_energy(fragment.to_pauli_sum(), config)
    return energies


def build_interaction_graph(
    patient: Patient, energies: dict[tuple[int, int], float], feature_dim: int
) -> InteractionGraph:
    n_nodes = len(patient.agents)
    features = np.stack([node_feature(m, feature_dim) for m in patient.agents])
    edge_energy = np.array([energies[pair] for pair in patient.pairs], dtype=np.float64)
    relations = np.array(patient.relations, dtype=np.int64)
    return InteractionGraph(
        n_nodes=n_nodes,
        edges=patient.pairs,
        node_features=features,
        edge_relations=relations,
        energies=edge_energy,
    )


def coupling_map(graph: InteractionGraph, scale: float = 1.0) -> dict[tuple[int, int], float]:
    return {
        edge: coupling_from_energy(float(graph.energies[index]), scale)
        for index, edge in enumerate(graph.edges)
    }
