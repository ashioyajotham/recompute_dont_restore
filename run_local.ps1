<#
.SYNOPSIS
  Local (CPU / Windows) quick start for "Recompute, Don't Store".

.DESCRIPTION
  Sets up a Python venv, installs CPU JAX, runs the CPU-safe test slice,
  executes the naive-JAX baseline benchmark, runs a small forward-pass
  smoke test against the naive reference, and launches Jupyter pointed
  at the three story notebooks.

  The TPU-gated tests (tests/test_flash_attention.py and the Pallas
  half of tests/test_gqa.py) auto-skip on CPU — run those on Colab
  via run_colab.sh.

.PARAMETER SkipInstall
  Reuse an existing .venv without reinstalling dependencies.

.PARAMETER SkipNotebooks
  Do everything except the final "jupyter notebook" launch.

.PARAMETER Baseline
  Run 02_naive_jax_baseline/benchmark_baseline.py (slow on CPU for long
  sequences; defaults to off).

.EXAMPLE
  ./run_local.ps1
  ./run_local.ps1 -SkipInstall
  ./run_local.ps1 -Baseline -SkipNotebooks
#>

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipNotebooks,
    [switch]$Baseline
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Section($msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. venv + CPU JAX
# ---------------------------------------------------------------------------
Section "1. Python venv + CPU JAX"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

if (-not $SkipInstall) {
    python -m pip install --upgrade pip | Out-Null
    # requirements.txt pins jax[tpu] which has no Windows wheel; install CPU
    # JAX directly instead. The rest of the deps are portable.
    pip install "jax[cpu]" "jaxlib" "numpy>=1.24" "matplotlib>=3.7" pytest jupyter
}

python -c "import jax; print('JAX', jax.__version__, '->', jax.devices())"

# ---------------------------------------------------------------------------
# 2. CPU-safe pytest slice
# ---------------------------------------------------------------------------
Section "2. CPU-safe tests (TPU tests auto-skip)"

# Everything under tests/ — @pytest.mark.tpu tests are skipped by conftest.
pytest -v

# ---------------------------------------------------------------------------
# 3. Smoke test: naive kernel vs a hand-rolled NumPy reference
# ---------------------------------------------------------------------------
Section "3. Smoke test: naive attention forward pass"

$smokeTest = @'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "02_naive_jax_baseline"))

import jax
import jax.numpy as jnp
import numpy as np
from standard_attention import attention

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
shape = (1, 2, 256, 128)
q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)

out = attention(q, k, v)
print(f"output shape: {out.shape}, dtype: {out.dtype}")
arr = np.asarray(out, dtype=np.float32)
assert np.all(np.isfinite(arr)), "non-finite values in output"
print(f"output summary: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")

grads = jax.grad(lambda q, k, v: attention(q, k, v).sum(), argnums=(0, 1, 2))(q, k, v)
for name, g in zip(("dq", "dk", "dv"), grads):
    ok = np.all(np.isfinite(np.asarray(g)))
    print(f"{name}: shape={g.shape} finite={ok}")

print("OK: naive forward + gradients sane.")
'@
$smokeTest | python -

# ---------------------------------------------------------------------------
# 4. Optional: naive baseline sequence-length sweep
# ---------------------------------------------------------------------------
if ($Baseline) {
    Section "4. Naive baseline sweep (CPU — can be slow)"
    python 02_naive_jax_baseline\benchmark_baseline.py --plot `
        --out 05_benchmarks\results\baseline_cpu.json
    Write-Host "Results -> 05_benchmarks\results\baseline_cpu.*" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 5. Launch notebooks
# ---------------------------------------------------------------------------
if (-not $SkipNotebooks) {
    Section "5. Launching Jupyter"
    Write-Host "Recommended reading order:"
    Write-Host "  1. 00_motivation\attention_complexity.ipynb"
    Write-Host "  2. 01_flash_attention_math\tiling_and_online_softmax.ipynb"
    Write-Host "  3. 06_splash_attention_comparison\splash_vs_flash.ipynb"
    Write-Host ""
    jupyter notebook
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
