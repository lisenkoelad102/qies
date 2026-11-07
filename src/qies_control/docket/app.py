from pathlib import Path

from cyclopts import App

from qies_control.atelier.settings import load_experiment
from qies_control.docket.evaluate import run_evaluation
from qies_control.docket.export import export_head
from qies_control.docket.infer import interaction_energy
from qies_control.docket.train import run_training

app = App(
    name="qies-control",
    help="Quantum interaction-energy-shaped closed-loop control for Crohn's care.",
)


@app.command
def train(
    config: Path | None = None,
    override: list[str] | None = None,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> None:
    experiment = load_experiment(config, override)
    artifacts = run_training(experiment, seed=seed, checkpoint_path=checkpoint)
    print(f"parity report: {artifacts.parity_report}")
    print(f"head loss: {artifacts.head_history[0]:.4f} -> {artifacts.head_history[-1]:.4f}")
    print(
        f"control loss: {artifacts.control_history[0]:.4f} -> {artifacts.control_history[-1]:.4f}"
    )


@app.command
def evaluate(
    config: Path | None = None,
    override: list[str] | None = None,
    episodes: int = 4,
) -> None:
    experiment = load_experiment(config, override)
    summary = run_evaluation(experiment, episodes=episodes)
    print(summary)


@app.command
def infer(
    left: str,
    right: str,
    config: Path | None = None,
    override: list[str] | None = None,
) -> None:
    experiment = load_experiment(config, override)
    print(interaction_energy(left, right, experiment.vqe))


@app.command
def export(
    config: Path | None = None,
    override: list[str] | None = None,
    output: Path = Path("qies_head.onnx"),
) -> None:
    experiment = load_experiment(config, override)
    path = export_head(experiment, output)
    print(f"exported fusion head to {path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
