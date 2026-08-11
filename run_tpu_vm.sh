#!/usr/bin/env bash
set -euo pipefail

TPU_VERSION="${TPU_VERSION:-v5e}"
SKIP_BENCH="${SKIP_BENCH:-0}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e . pytest matplotlib

python -m pytest -m tpu -v
if [[ "${SKIP_BENCH}" != "1" ]]; then
  python 05_benchmarks/memory_profile.py
  python 05_benchmarks/throughput_tflops.py --tpu "${TPU_VERSION}"
fi
