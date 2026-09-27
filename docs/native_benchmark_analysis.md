# Offline native benchmark analysis

Source: three fresh-process BF16 B1/H1/D128 benchmark reports from the private archive with SHA-256 `489797e795f50c8d3d1804bcb3e5c521365879a1fc068f5e3f7943470c9267a1`. All cases passed the recorded `atol=rtol=0.05` output/gradient checks. Each method has five warmups and 30 synchronized trials per process on one v5e chip.

The table gives the **median of three per-process p50s** in ms (min–max across processes), followed by paired Pallas/naive p50 ratios. Forward+backward means the complete loss/gradient call, not isolated backward.

| S | Mask | Naive F | Pallas F | F ratio | Naive F+B | Pallas F+B | F+B ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | noncausal | 0.153 (0.136–0.155) | 0.154 (0.136–0.156) | 1.002 (0.999–1.002)× | 0.210 (0.184–0.212) | 0.213 (0.184–0.213) | 1.004 (1.002–1.013)× |
| 256 | causal | 0.153 (0.133–0.156) | 0.153 (0.133–0.156) | 0.999 (0.997–1.002)× | 0.212 (0.182–0.214) | 0.213 (0.183–0.214) | 1.008 (1.002–1.008)× |
| 512 | noncausal | 0.152 (0.135–0.157) | 0.156 (0.138–0.159) | 1.023 (1.013–1.026)× | 0.211 (0.185–0.213) | 0.228 (0.203–0.231) | 1.081 (1.080–1.101)× |
| 512 | causal | 0.154 (0.149–0.156) | 0.157 (0.152–0.159) | 1.018 (1.014–1.019)× | 0.214 (0.207–0.216) | 0.232 (0.226–0.233) | 1.082 (1.079–1.094)× |
| 1024 | noncausal | 0.154 (0.148–0.156) | 0.185 (0.179–0.188) | 1.203 (1.198–1.211)× | 0.227 (0.221–0.227) | 0.293 (0.288–0.296) | 1.303 (1.293–1.305)× |
| 1024 | causal | 0.152 (0.148–0.154) | 0.186 (0.179–0.186) | 1.211 (1.202–1.226)× | 0.225 (0.220–0.227) | 0.296 (0.292–0.299) | 1.327 (1.304–1.327)× |
| 2048 | noncausal | 0.168 (0.164–0.168) | 0.290 (0.286–0.291) | 1.727 (1.726–1.745)× | 0.267 (0.266–0.270) | 0.545 (0.543–0.548) | 2.045 (2.032–2.046)× |
| 2048 | causal | 0.168 (0.163–0.169) | 0.290 (0.285–0.291) | 1.727 (1.724–1.747)× | 0.270 (0.269–0.272) | 0.564 (0.560–0.565) | 2.079 (2.077–2.090)× |
| 4096 | noncausal | 0.233 (0.230–0.233) | 0.708 (0.704–0.711) | 3.050 (3.041–3.060)× | 0.456 (0.455–0.461) | 1.548 (1.543–1.548) | 3.396 (3.350–3.403)× |
| 4096 | causal | 0.233 (0.230–0.235) | 0.705 (0.700–0.705) | 3.031 (2.994–3.049)× | 0.463 (0.460–0.465) | 1.614 (1.610–1.617) | 3.490 (3.469–3.497)× |

## Tail latency and first calls

The following are medians of the three per-process p95s and first-call host wall times, respectively. The first call **may include compilation and cache effects**; it is not isolated compiler time.

| S | Mask | Naive F p95 | Pallas F p95 | Naive F+B p95 | Pallas F+B p95 | Naive F first | Pallas F first | Naive F+B first | Pallas F+B first |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | noncausal | 0.164 | 0.166 | 0.223 | 0.222 | 172.798 | 109.656 | 216.687 | 186.038 |
| 256 | causal | 0.162 | 0.164 | 0.221 | 0.226 | 183.036 | 116.219 | 229.572 | 200.353 |
| 512 | noncausal | 0.158 | 0.169 | 0.223 | 0.235 | 364.860 | 106.537 | 411.655 | 181.879 |
| 512 | causal | 0.164 | 0.164 | 0.226 | 0.241 | 373.367 | 118.252 | 421.742 | 199.289 |
| 1024 | noncausal | 0.157 | 0.193 | 0.239 | 0.302 | 909.826 | 107.015 | 962.492 | 223.157 |
| 1024 | causal | 0.161 | 0.192 | 0.232 | 0.304 | 909.864 | 116.389 | 962.909 | 206.703 |
| 2048 | noncausal | 0.175 | 0.296 | 0.281 | 0.552 | 942.685 | 108.705 | 3966.372 | 185.263 |
| 2048 | causal | 0.176 | 0.297 | 0.281 | 0.575 | 950.326 | 151.889 | 2141.458 | 206.114 |
| 4096 | noncausal | 0.238 | 0.715 | 0.469 | 1.558 | 941.048 | 109.909 | 996.305 | 195.458 |
| 4096 | causal | 0.240 | 0.715 | 0.474 | 1.623 | 949.228 | 118.598 | 1018.767 | 210.236 |

## What this establishes

- Maximum recorded absolute errors across cases/repetitions: o=0.019531, dq=0.015625, dk=0.015625, dv=0.015625.
- The S=4096 slowdown persists across all three repetitions for both masks and both complete call types. At S=256, differences are small relative to the host-dispatch timing floor.
- These are sequential, fixed-order host wall timings: naive forward, Pallas forward, naive forward+backward, Pallas forward+backward. They are not device-only timings; order and cache effects were not controlled.
- Do not subtract medians to infer backward-only time. No per-operation peak-HBM, FLOP utilization, or causal executed-work reduction was measured.
- Next paid experiment, if warranted: profile matched S=1024 and S=4096 cases with synchronized device traces and counterbalanced method order before changing kernel logic.
