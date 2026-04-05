# xprof Guide for Flash Attention Kernels

This guide covers capturing and interpreting xprof traces for the Pallas
kernels in this project.

## Capturing a trace

```python
import jax
from pathlib import Path

profile_dir = "/tmp/xprof_flash_attention"
Path(profile_dir).mkdir(parents=True, exist_ok=True)

with jax.profiler.trace(profile_dir):
    jax.block_until_ready(flash_attention_forward(q, k, v))
```

Then open in TensorBoard:

```bash
tensorboard --logdir /tmp/xprof_flash_attention
```

Navigate to the **Profile** tab -> **Trace Viewer**.

## What to look for

### Well-utilized TPU

- `MXU` (Matrix Unit) tiles appear dense and continuous with minimal gaps
- `HBM` reads appear in bursts matching the KV tile prefetch pattern
- The `DMA` channel shows the KV tile load overlapping with the previous tile's
  compute — this is Mosaic's pipeline prefetching at work

### Pathological patterns

| Pattern | Likely cause |
|---|---|
| MXU bubbles (gaps between tiles) | `dimension_semantics` wrong — Mosaic can't pipeline |
| VMEM pressure / spills | Block sizes too large; reduce via `get_block_sizes()` |
| Sequential HBM reads (no overlap) | Missing or wrong `dimension_semantics="arbitrary"` on KV axis |
| Very short MXU ops | Block sizes too small; tiles below `MIN_BLOCK_SIZE=128` |

## Reading the op names

In the trace viewer, Pallas ops appear under names like:

```
flash_fwd_kernel[b=0,h=0,q_tile=0,kv_tile=3]
```

The `kv_tile` index increments sequentially within a fixed `(b, h, q_tile)`.
You should see the `DMA load K/V` for tile `j+1` overlapping with the `MXU`
compute for tile `j`.

## Memory timeline

In the **Memory Profile** sub-tab:
- Naive attention: the allocation line jumps sharply at the point the full
  attention matrix `(batch, heads, seq, seq)` is materialized
- Flash attention: no such jump — allocation stays flat across the KV loop

## Compute utilization numbers

From the **Overview** page, read:
- **TPU idle time**: should be < 5% for well-tuned kernels
- **Infeed/outfeed time**: HBM transfer time; high values indicate memory-bound
  operation (expected at small sequence lengths)

## Correlating with benchmark numbers

If your MFU (from `throughput_tflops.py`) is low:

1. Check the trace for MXU bubbles — fix with correct `dimension_semantics`
2. Check VMEM spills — fix with smaller block sizes
3. Check if the kernel is HBM-bandwidth bound rather than compute-bound —
   this is expected at seq < 2048; at longer sequences the kernel should
   become compute-bound

## Comparing naive vs flash in the same trace

```python
with jax.profiler.trace(profile_dir):
    jax.block_until_ready(naive_attention(q, k, v))
    jax.block_until_ready(flash_attention_forward(q, k, v))
```

In the trace viewer, the naive attention op will show a single large allocation
and compute block. The flash attention op will show many smaller, pipelined
compute blocks.
