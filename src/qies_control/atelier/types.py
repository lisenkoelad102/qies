from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Molecule:
    symbols: tuple[str, ...]
    bonds: tuple[tuple[int, int, float], ...]

    @property
    def n_heavy(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class AgentPair:
    left: str
    right: str
    left_molecule: Molecule
    right_molecule: Molecule
    nutrient: bool


@dataclass(frozen=True)
class InteractionGraph:
    n_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_features: FloatArray
    edge_relations: IntArray
    energies: FloatArray


@dataclass(frozen=True)
class ParitySample:
    graph: InteractionGraph
    label: IntArray
    protocol: str
