import torch
from torch import Tensor
from torch.nn import functional as F


def protocol_loss(logits: Tensor, target: Tensor, protocol: str) -> Tensor:
    if protocol == "relation":
        return F.cross_entropy(logits.unsqueeze(0), target.reshape(1))
    if protocol == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, target.float())
    if protocol == "binary":
        return F.binary_cross_entropy_with_logits(logits, target.float())
    raise ValueError(f"unknown protocol: {protocol}")


def predict_label(logits: Tensor, protocol: str) -> Tensor:
    if protocol == "relation":
        return torch.argmax(logits, dim=-1, keepdim=True)
    if protocol in {"multilabel", "binary"}:
        return (torch.sigmoid(logits) > 0.5).long()
    raise ValueError(f"unknown protocol: {protocol}")
