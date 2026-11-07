from typing import cast

import torch
from torch import Tensor, nn

from qies_control.atelier.types import InteractionGraph
from qies_control.basting.backbone import RelationBackbone, graph_to_tensors


class EnergyGraphHead(nn.Module):
    def __init__(self, hidden: int, n_labels: int, use_energy: bool = True) -> None:
        super().__init__()
        self.use_energy = use_energy
        in_dim = hidden + (1 if use_energy else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, embedding: Tensor, energy: Tensor) -> Tensor:
        if self.use_energy:
            features = torch.cat([embedding, energy.reshape(1)], dim=-1)
        else:
            features = embedding
        return cast(Tensor, self.net(features))


class ParityModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        n_relations: int,
        n_labels: int,
        use_energy: bool = True,
        use_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.use_fusion = use_fusion
        self.backbone = RelationBackbone(feature_dim, hidden, n_relations)
        self.head = EnergyGraphHead(hidden, n_labels, use_energy=use_energy)

    def forward(self, graph: InteractionGraph) -> Tensor:
        features, edges, relations, energies = graph_to_tensors(graph)
        embedding = self.backbone(features, edges, relations, energies)
        energy_scalar = energies.mean() if self.use_fusion else torch.zeros(1).squeeze(0)
        return cast(Tensor, self.head(embedding, energy_scalar))
