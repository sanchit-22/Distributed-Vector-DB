#!/usr/bin/env bash
# Downloads ANN benchmark datasets as HDF5 files from ann-benchmarks.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p sift10k sift1m

echo "Downloading SIFT1M dataset (HDF5, ~500 MB)..."
wget -c http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O sift1m/sift1m.hdf5

# SIFT10K is a slice we extract from SIFT1M at runtime via load_dataset.py
echo "Done. Use load_dataset.py to load SIFT10K as a slice of SIFT1M."
