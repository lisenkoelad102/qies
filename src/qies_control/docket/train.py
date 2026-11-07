from dataclasses import dataclass
from pathlib import Path

from qies_control.atelier.settings import ExperimentConfig


@dataclass
class TrainingArtifacts:
    parity_report: dict[str, float]
    head_history: list[float]
    control_history: list[float]
    potentials: list[float]


def run_training(
    experiment: ExperimentConfig, seed: int = 0, checkpoint_path: str | Path | None = None
) -> TrainingArtifacts:
    raise NotImplementedError("Training requires verified benchmark and patient data integration.")
