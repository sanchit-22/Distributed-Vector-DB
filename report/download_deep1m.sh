#!/usr/bin/env bash
# Download the ANN-Benchmarks DEEP 96D HDF5 dataset.
#
# The ANN-Benchmarks public file is named deep-image-96-angular.hdf5 and is
# about 3.6 GB. For this project report, Deep1M means the first 1,000,000 train
# vectors from that file. VecScaleDB evaluates them with L2 because the current
# engine is Faiss IVF_FLAT with L2 distance.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/deep1m

DEEP_URL="${DEEP_URL:-http://ann-benchmarks.com/deep-image-96-angular.hdf5}"
OUT="data/deep1m/deep1m.hdf5"

echo "Downloading Deep 96D HDF5 dataset to $OUT"
echo "Source: $DEEP_URL"
wget -c "$DEEP_URL" -O "$OUT"

echo "Done. Use the first 1,000,000 train vectors as Deep1M."

