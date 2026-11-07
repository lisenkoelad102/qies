import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base: int, offset: int) -> int:
    mixed = (base * 1_000_003 + offset * 2_654_435_761) & 0x7FFF_FFFF
    return int(mixed)
