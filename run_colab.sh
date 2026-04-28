#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Recompute, Don't Store -- Colab TPU quick start.
#
# Usage (paste into a cell on a TPU Colab runtime):
#     !git clone https://github.com/ashioyajotham/recompute_dont_restore.git
#     %cd recompute_dont_restore
#     !bash run_colab.sh
#
# By default, benchmark results are copied to /content/drive/MyDrive/
# recompute_dont_store_results/<timestamp>/ so you can pull them back to
# your local checkout later and commit them under 05_benchmarks/results/.
#
# Environment variables (all optional):
#   TPU_VERSION    v4 | v5e | v5p        (default: v5e)
#   SKIP_INSTALL   1 to reuse existing install
#   SKIP_TESTS     1 to skip pytest
#   SKIP_BENCH     1 to skip the benchmark sweep
#   SKIP_DRIVE     1 to keep results only in the Colab VM (no Drive mount)
#   RESULT_DIR     override local results directory
#                  (default: 05_benchmarks/results)
# ----------------------------------------------------------------------------

set -euo pipefail

TPU_VERSION="${TPU_VERSION:-v5e}"
RESULT_DIR="${RESULT_DIR:-05_benchmarks/results}"
TS="$(date +%Y%m%d_%H%M%S)"
DRIVE_DIR="/content/drive/MyDrive/recompute_dont_store_results/${TS}"

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
# 1. Install JAX + TPU backend
# ---------------------------------------------------------------------------
section "1. Install JAX (TPU)"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    pip install -q --upgrade pip
    # Colab TPU wheels live on libtpu_releases.
    pip install -q "jax[tpu]" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install -q "numpy>=1.24" "matplotlib>=3.7" pytest
fi

python - <<'PY'
import jax
devices = jax.devices()
print("JAX", jax.__version__)
print("devices:", devices)
has_tpu = any(d.platform == "tpu" for d in devices)
if not has_tpu:
    raise SystemExit(
        "No TPU device found. On Colab: Runtime -> Change runtime type -> TPU."
    )
print("OK: TPU visible.")
PY

# ---------------------------------------------------------------------------
# 2. Full test suite (TPU tests now execute)
# ---------------------------------------------------------------------------
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    section "2. pytest (full suite, TPU tests included)"
    pytest -v
fi

# ---------------------------------------------------------------------------
# 3. Correctness smoke: flash_fwd vs naive attention
# ---------------------------------------------------------------------------
section "3. Smoke: flash_fwd vs naive on bf16 (seq=512)"

python - <<'PY'
import sys
sys.path += ["03_pallas_kernels", "02_naive_jax_baseline"]

import jax, jax.numpy as jnp, numpy as np
from flash_fwd import flash_attention_forward
from standard_attention import attention as naive
from utils import get_block_sizes

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
shape = (1, 2, 512, 128)
q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)

bs = get_block_sizes(512, 128, jnp.bfloat16)
o_flash, m, l = flash_attention_forward(q, k, v, block_sizes=bs)
o_naive = naive(q, k, v)

max_abs = float(np.max(np.abs(
    np.asarray(o_flash, dtype=np.float32) - np.asarray(o_naive, dtype=np.float32)
)))
print(f"block sizes: block_q={bs.block_q} block_kv={bs.block_kv}")
print(f"max abs diff vs naive: {max_abs:.4e}  (expect < 5e-2 for bf16)")
assert max_abs < 5e-2, "flash output diverges from naive"
print("OK: flash forward matches naive reference.")
PY

# ---------------------------------------------------------------------------
# 4. Benchmark sweeps (naive baseline + memory + throughput)
# ---------------------------------------------------------------------------
if [[ "${SKIP_BENCH:-0}" != "1" ]]; then
    section "4. Benchmarks (baseline + memory + throughput / MFU)"
    mkdir -p "${RESULT_DIR}"

    python 02_naive_jax_baseline/benchmark_baseline.py --plot \
        --out "${RESULT_DIR}/baseline.json"

    python 05_benchmarks/memory_profile.py --plot \
        --out "${RESULT_DIR}/memory.json"

    python 05_benchmarks/throughput_tflops.py --tpu "${TPU_VERSION}" --plot \
        --out "${RESULT_DIR}/throughput_${TPU_VERSION}.json"
fi

# ---------------------------------------------------------------------------
# 5. Persist results to Drive so they survive the runtime
# ---------------------------------------------------------------------------
if [[ "${SKIP_DRIVE:-0}" != "1" ]]; then
    section "5. Copy results to Google Drive"

    python - <<'PY'
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("Drive mounted.")
except Exception as e:
    print(f"WARNING: could not mount Drive ({e}).")
    print("Results will stay in the Colab VM at ${RESULT_DIR}.")
    raise SystemExit(0)
PY

    if [[ -d /content/drive/MyDrive ]]; then
        mkdir -p "${DRIVE_DIR}"
        cp -r "${RESULT_DIR}"/* "${DRIVE_DIR}/" 2>/dev/null || true
        echo ""
        echo "Results copied to:"
        echo "  ${DRIVE_DIR}"
        echo ""
        echo "To pull them back to your local checkout later:"
        echo "  - Download the folder from Drive"
        echo "  - Drop the files into ${RESULT_DIR}/ in your local repo"
        echo "  - git add -f ${RESULT_DIR}/*.json ${RESULT_DIR}/*.png"
        echo "    (the .gitignore excludes this directory by default)"
    fi
fi

section "Done"
echo "Local (in-VM) results: ${RESULT_DIR}"
[[ "${SKIP_DRIVE:-0}" != "1" ]] && echo "Drive backup:          ${DRIVE_DIR}"
