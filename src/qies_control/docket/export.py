from pathlib import Path

import torch

from qies_control.atelier.settings import ExperimentConfig
from qies_control.basting.fusion_head import EnergyGraphHead
from qies_control.pressing.benchmark import HIDDEN


def export_head(experiment: ExperimentConfig, output: str | Path) -> str:
    head = EnergyGraphHead(HIDDEN, experiment.cohort.n_labels, use_energy=True)
    head.eval()
    embedding = torch.zeros(HIDDEN, dtype=torch.float32)
    energy = torch.zeros(1, dtype=torch.float32)
    torch.onnx.export(
        head,
        (embedding, energy),
        str(output),
        input_names=["embedding", "energy"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    return str(output)
