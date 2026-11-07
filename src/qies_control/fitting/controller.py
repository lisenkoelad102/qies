import math
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from qies_control.atelier.settings import ControllerConfig
from qies_control.atelier.types import FloatArray

_LOG_2PI = math.log(2.0 * math.pi)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, config: ControllerConfig) -> None:
        super().__init__()
        hidden = config.hidden
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.value = nn.Linear(hidden, 1)
        self.register_buffer("low", torch.tensor([config.interval_low, 0.0], dtype=torch.float32))
        self.register_buffer(
            "span",
            torch.tensor([config.interval_high - config.interval_low, 1.0], dtype=torch.float32),
        )

    def _bound(self, raw: Tensor) -> Tensor:
        low = cast(Tensor, self.low)
        span = cast(Tensor, self.span)
        return low + span * torch.sigmoid(raw)

    def _statistics(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.body(state)
        mean = self.mean(latent)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        value = self.value(latent).squeeze(-1)
        return mean, std, value

    def act(self, state: FloatArray) -> tuple[FloatArray, Tensor, Tensor, Tensor]:
        tensor_state = torch.tensor(state, dtype=torch.float32)
        mean, std, value = self._statistics(tensor_state)
        noise = torch.randn(2)
        raw = (mean + std * noise).detach()
        log_prob = (-0.5 * ((raw - mean) / std) ** 2 - torch.log(std) - 0.5 * _LOG_2PI).sum(-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + torch.log(std)).sum(-1)
        action = self._bound(raw)
        result: FloatArray = action.detach().numpy().astype(np.float64)
        return result, log_prob, value, entropy

    def greedy(self, state: FloatArray) -> FloatArray:
        tensor_state = torch.tensor(state, dtype=torch.float32)
        mean, _, _ = self._statistics(tensor_state)
        action = self._bound(mean)
        result: FloatArray = action.detach().numpy().astype(np.float64)
        return result
