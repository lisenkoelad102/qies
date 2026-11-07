# QIES-Control

QIES-Control provides research components for drug-interaction graph analysis, quantum-inspired energy calculations, and treatment-control modelling.

## Installation

```bash
pip install -e ".[dev]"
```

## Data and execution

Benchmark records, patient data, and trained weights are not included. `cloth.datasets.load_benchmark` reads caller-provided benchmark manifests. The command-line training and evaluation workflows require verified data integration and currently stop with an error. The manuscript's reported results have not been reproduced using this repository.

The quantum code performs statevector calculations on classical hardware. The fragment Hamiltonian uses structured approximations based on molecular graphs; its output is not a validated measurement of drug interactions. The treatment model and controller require independent validation.

`cloth` contains data structures and graph utilities; `draft` and `canvas` contain Hamiltonian and quantum calculation components; `basting` contains graph models; `fitting` contains controller components; `notions` and `pressing` contain evaluation utilities. Configuration files are in `configs/`.

Potential sources and access links are listed by `scripts/prepare_data.sh`. Each source's access and usage terms apply.

```bash
pytest -q
```

This is research software and has not been validated for clinical use.
