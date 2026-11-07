from collections.abc import Callable
from typing import cast

import torch


def descend(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    optimizer.zero_grad()
    cast(Callable[[], None], loss.backward)()
    optimizer.step()
