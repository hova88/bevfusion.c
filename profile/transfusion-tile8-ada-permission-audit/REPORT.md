# TransFusion tile-8 Ada permission audit

## Question

Can source-attributed Nsight Compute counters be collected for the promoted
strict eight-query TransFusion attention kernel on the real production shape?

## Setup

- GPU: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9
- Driver: 610.62
- CUDA compiler: 12.4
- Nsight Compute CLI: 2024.1.1.0
- Source: [`src/cuda_transfusion.cu`](../../src/cuda_transfusion.cu)
- Build: `make build/test_cuda_transfusion CUDA_ARCH=sm_89`
- Build flags: `-O3 -lineinfo -arch=sm_89`
- Workload: exported production model plus the saved TransFusion oracle

Command:

```sh
ncu --section SpeedOfLight \
  -k 'regex:flash_attention_query_tile_kernel' -c 1 \
  -o profile/transfusion-tile8-ada-permission-audit/reports/speed-of-light \
  build/test_cuda_transfusion bevfusion.bfw build/transfusion_decoder_oracle.bfw
```

## Result

**Observed:** the workload and oracle completed, including `1.00e-5` maximum
center error, but NCU returned `ERR_NVGPUCTRPERM` and reported that no kernels
were profiled. It created no `.ncu-rep` file.

**Conclusion:** no Speed-of-Light, occupancy, memory, stall, or source-line
metric can be claimed from this run. An administrator must enable NVIDIA
performance-counter access before the full and source collections described in
the [performance workflow](../../wiki/09-performance-workflow.md) can proceed.
