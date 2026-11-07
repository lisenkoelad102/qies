import numpy as np
import torch
from torch import Tensor, nn

from qies_control.atelier.types import InteractionGraph


def graph_to_tensors(graph: InteractionGraph) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    features = torch.tensor(graph.node_features, dtype=torch.float32)
    if graph.edges:
        edges = torch.tensor(np.array(graph.edges), dtype=torch.long)
    else:
        edges = torch.zeros((0, 2), dtype=torch.long)
    relations = torch.tensor(graph.edge_relations, dtype=torch.long)
    energies = torch.tensor(graph.energies, dtype=torch.float32)
    return features, edges, relations, energies


class RelationBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, n_relations: int, n_layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.embed = nn.Linear(feature_dim, hidden)
        self.relation = nn.Embedding(max(n_relations, 1), hidden)
        self.message = nn.Linear(2 * hidden + 1, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.readout = nn.Linear(hidden, hidden)

    def forward(
        self, features: Tensor, edges: Tensor, relations: Tensor, energies: Tensor
    ) -> Tensor:
        h = torch.tanh(self.embed(features))
        for _ in range(self.n_layers):
            aggregate = torch.zeros_like(h)
            if edges.shape[0] > 0:
                source = edges[:, 0]
                target = edges[:, 1]
                relation_bias = self.relation(relations.clamp(max=self.relation.num_embeddings - 1))
                message_input = torch.cat([h[source], h[target], energies.unsqueeze(-1)], dim=-1)
                message = torch.tanh(self.message(message_input)) + relation_bias
                aggregate = aggregate.index_add(0, target, message)
                aggregate = aggregate.index_add(0, source, message)
            h = self.update(aggregate, h)
        pooled = h.mean(dim=0, keepdim=True)
        return torch.tanh(self.readout(pooled)).squeeze(0)
