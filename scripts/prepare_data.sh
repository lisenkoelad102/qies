#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/raw}"
mkdir -p "${ROOT}"

cat <<'NOTE'
Benchmark sources (each requires registration or bulk download; none are redistributed here):

  DrugBank-86        https://go.drugbank.com/releases/latest      (Academic / Non-commercial)
  TWOSIDES-200       https://nsides.io/                           (Open download)
  BioSNAP ChCh-Miner https://snap.stanford.edu/biodata/           (Open research use)
  ChChSe-Decagon     https://snap.stanford.edu/biodata/           (Open research use)
  QM9 geometries     http://quantum-machine.org/datasets/         (CC0)
  PubChem structures https://pubchem.ncbi.nlm.nih.gov/            (Public domain)

Place a per-benchmark manifest as ${ROOT}/<name>.npz with arrays {features, energies, labels}
to feed qies_control.cloth.datasets.load_benchmark. Training and evaluation also require
verified patient data and integration with the chosen sources.
NOTE
