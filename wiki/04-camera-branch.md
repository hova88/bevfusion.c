# Camera branch

The camera path combines a strict Swin-T backbone, a cuDNN FPN/depth neck, and
a deterministic lift-splat plan. Its optimization story is chiefly about
bounded residency and removing atomic geometry work.

## Swin candidate ladder

| Candidate | Warm time | Residency | Decision |
|---|---:|---:|---|
| One query per attention block | 87.749 ms | 393.73 MiB | Baseline |
| Eight-query tile | 71.336 ms | 393.73 MiB | Intermediate |
| Sixteen-query tile | 75.229 ms | 393.73 MiB | Reject regression |
| Shared K/V | 63.652 ms | 393.73 MiB | Intermediate |
| Shared K/V, bounded FFN/windows | 58.055 ms | 216.27 MiB | Promote |
| TF32 | not promoted | — | Reject: `1.87e-3` max error |

All rows are **measured** production-shape values from
[`cuda_swin.txt`](../runs/rtx4060ti-2026-07-18/cuda_swin.txt). The promoted
strict route reduces warm latency by 33.8% and recorded residency by 45.1%
relative to the first layout. The real-weight maximum absolute error is
`2.86e-6`, including a forced chunk boundary crossing a shifted mask.

## Lift-splat changed the algorithm

The original route atomically accumulated lifted samples and measured 85.577
ms warm. Precomputing ranks while retaining atomics regressed to 92.977 ms.
Stable cell intervals removed the atomics: NCHW context measured 5.549 ms and
NHWC context measured 1.389 ms. Direct calibration-to-interval preparation
adds 0.363 ms per changed calibration, for a promoted 1.752 ms total.

This is a causal A/B result, not a profiler-counter diagnosis. NCU counters
were unavailable on the host, so the wiki does not claim an atomic stall
percentage or bandwidth ceiling. The implementation is in
[`src/cuda_lss.cu`](../src/cuda_lss.cu); the full ladder and oracle values are
recorded in [`cuda_lss_intervals.txt`](../runs/rtx4060ti-2026-07-18/cuda_lss_intervals.txt).

**Measured:** the composed camera branch reaches 67.007 ms warm, 472.95 MiB
recorded total, zero inference transfer, and bit-exact repeated image-BEV
output. Its composed oracle maxima are `8.11e-6` for logits and `2.38e-6` for
context.
