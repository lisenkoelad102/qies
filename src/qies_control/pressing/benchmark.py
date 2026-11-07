import numpy as np
import torch
from torch import optim

from qies_control.atelier.optim import descend
from qies_control.atelier.seeding import set_seed
from qies_control.atelier.settings import ExperimentConfig
from qies_control.atelier.types import ParitySample
from qies_control.basting.fusion_head import ParityModel
from qies_control.basting.losses import predict_label, protocol_loss
from qies_control.notions.classification import classification_report
from qies_control.notions.ranking import ranking_report

HIDDEN = 32

REFERENCE_TABLE_I: dict[str, float] = {
    "drugbank86_macro_f1": 0.9140,
    "drugbank86_accuracy": 0.9300,
    "drugbank86_cohen_kappa": 0.9160,
    "twosides200_auroc": 0.9520,
    "twosides200_auprc": 0.9390,
}


def build_model(experiment: ExperimentConfig, use_energy: bool, use_fusion: bool) -> ParityModel:
    cohort = experiment.cohort
    return ParityModel(
        feature_dim=cohort.feature_dim,
        hidden=HIDDEN,
        n_relations=max(cohort.n_labels, 1),
        n_labels=cohort.n_labels,
        use_energy=use_energy,
        use_fusion=use_fusion,
    )


def train_parity(
    samples: list[ParitySample],
    experiment: ExperimentConfig,
    use_energy: bool = True,
    use_fusion: bool = True,
    seed: int = 0,
) -> tuple[ParityModel, list[float]]:
    set_seed(seed)
    model = build_model(experiment, use_energy, use_fusion)
    optimizer = optim.Adam(model.parameters(), lr=experiment.head_lr)
    order = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for _ in range(experiment.head_epochs):
        rng.shuffle(order)
        running = 0.0
        for index in order:
            sample = samples[index]
            logits = model(sample.graph)
            target = torch.tensor(sample.label, dtype=torch.long)
            loss = protocol_loss(logits, target, sample.protocol)
            descend(optimizer, loss)
            running += float(loss.item())
        history.append(running / max(len(samples), 1))
    return model, history


def evaluate_parity(model: ParityModel, samples: list[ParitySample]) -> dict[str, float]:
    protocol = samples[0].protocol
    model.eval()
    with torch.no_grad():
        if protocol == "relation":
            targets = np.array([int(s.label[0]) for s in samples], dtype=np.int64)
            predictions = np.array(
                [int(predict_label(model(s.graph), protocol).item()) for s in samples],
                dtype=np.int64,
            )
            return classification_report(targets, predictions)
        target_rows = np.stack([s.label for s in samples]).astype(np.int64)
        score_rows = np.stack([torch.sigmoid(model(s.graph)).numpy() for s in samples]).astype(
            np.float64
        )
        return ranking_report(target_rows, score_rows)


def benchmark_parity(
    experiment: ExperimentConfig,
    training_samples: list[ParitySample],
    evaluation_samples: list[ParitySample],
    seed: int = 0,
) -> dict[str, float]:
    if not training_samples or not evaluation_samples:
        raise ValueError("Training and evaluation samples must be provided.")
    model, _ = train_parity(
        training_samples, experiment, experiment.use_energy, experiment.use_fusion, seed
    )
    return evaluate_parity(model, evaluation_samples)
