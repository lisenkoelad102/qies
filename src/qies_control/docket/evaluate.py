from qies_control.atelier.settings import ExperimentConfig


def run_evaluation(experiment: ExperimentConfig, episodes: int = 4) -> dict[str, object]:
    raise NotImplementedError("Evaluation requires independent benchmark and patient data.")
