from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qies_control.atelier.settings import VqeConfig
from qies_control.atelier.types import (
    InteractionGraph,
    ParitySample,
)
from qies_control.cloth.graph import Patient, precompute_energies


@dataclass(frozen=True)
class BenchmarkSource:
    name: str
    version: str
    license_name: str
    url: str
    protocol: str
    n_labels: int


BENCHMARKS: dict[str, BenchmarkSource] = {
    "drugbank86": BenchmarkSource(
        "DrugBank-86",
        "DrugBank 6.0 / 86-relation split",
        "Academic / Non-commercial",
        "https://go.drugbank.com/releases/latest",
        "relation",
        86,
    ),
    "twosides200": BenchmarkSource(
        "TWOSIDES-200",
        "nSIDES release",
        "Open download",
        "https://nsides.io/",
        "multilabel",
        200,
    ),
    "biosnap": BenchmarkSource(
        "BioSNAP ChCh-Miner",
        "ChCh-Miner network",
        "Open research use",
        "https://snap.stanford.edu/biodata/",
        "binary",
        1,
    ),
    "decagon": BenchmarkSource(
        "ChChSe-Decagon",
        "Decagon side-effect network",
        "Open research use",
        "https://snap.stanford.edu/biodata/",
        "binary",
        1,
    ),
}


def patient_energies(
    patients: list[Patient], vqe_config: VqeConfig
) -> list[dict[tuple[int, int], float]]:
    return [precompute_energies(patient, vqe_config) for patient in patients]


def load_benchmark(name: str, root: str | Path) -> list[ParitySample]:
    source = BENCHMARKS[name]
    manifest = Path(root) / f"{name}.npz"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{source.name} manifest not found at {manifest}; obtain it from {source.url}"
        )
    payload = np.load(manifest, allow_pickle=True)
    features = payload["features"]
    energies = payload["energies"]
    labels = payload["labels"]
    samples: list[ParitySample] = []
    for index in range(features.shape[0]):
        graph = InteractionGraph(
            n_nodes=2,
            edges=((0, 1),),
            node_features=features[index],
            edge_relations=np.array([0], dtype=np.int64),
            energies=np.array([energies[index]], dtype=np.float64),
        )
        samples.append(
            ParitySample(
                graph=graph, label=labels[index].astype(np.int64), protocol=source.protocol
            )
        )
    return samples
