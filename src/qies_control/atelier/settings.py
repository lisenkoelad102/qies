from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class CohortConfig:
    n_patients: int = 64
    n_agents: int = 5
    feature_dim: int = 8
    protocol: str = "relation"
    n_labels: int = 86
    edge_density: float = 0.6
    seed: int = 0


@dataclass
class TwinConfig:
    horizon: int = 48
    dt: float = 1.0
    therapeutic_low: float = 3.0
    therapeutic_high: float = 7.0
    clearance: float = 0.18
    volume: float = 3.2
    healing_rate: float = 0.05
    damage_gain: float = 0.12
    nutrition_gain: float = 0.08
    burden_sensitivity: float = 0.9
    seed: int = 0


@dataclass
class VqeConfig:
    n_qubits: int = 8
    depth: int = 12
    spsa_iters: int = 60
    spsa_a: float = 0.12
    spsa_c: float = 0.1
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    seed: int = 0


@dataclass
class QaoaConfig:
    p_layers: int = 2
    opt_iters: int = 80
    step: float = 0.15
    seed: int = 0


@dataclass
class ControllerConfig:
    gamma: float = 0.97
    shaping_c: float = 1.0
    interaction_penalty: float = 0.5
    hidden: int = 64
    lr: float = 3e-4
    episodes: int = 200
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    interval_low: float = 4.0
    interval_high: float = 8.0
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "main"
    head_epochs: int = 20
    head_lr: float = 1e-3
    batch_size: int = 16
    precision: str = "fp32"
    use_energy: bool = True
    use_shaping: bool = True
    use_fusion: bool = True
    use_qaoa_aggregation: bool = True
    classical_proxy: bool = False
    eval_seeds: int = 20
    cohort: CohortConfig = field(default_factory=CohortConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    vqe: VqeConfig = field(default_factory=VqeConfig)
    qaoa: QaoaConfig = field(default_factory=QaoaConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


def load_experiment(
    path: str | Path | None = None, overrides: Sequence[str] | None = None
) -> ExperimentConfig:
    merged = OmegaConf.structured(ExperimentConfig)
    if path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(Path(path)))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    resolved = OmegaConf.to_object(merged)
    if not isinstance(resolved, ExperimentConfig):
        raise TypeError("resolved configuration is not an ExperimentConfig")
    return resolved
