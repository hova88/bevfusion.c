# Performance workflow

The repository uses an evidence ladder inspired by Kernel Design Agents:
freeze the contract, validate a baseline, measure one causal change, keep
negative results, and promote only candidates that pass the same oracle.

## Evidence labels

- **Measured:** a recorded timer, byte counter, error gate, or profiler value.
- **Calculated:** arithmetic derived from recorded values.
- **Observed:** a reproducible qualitative behavior, such as bit-exact repeat.
- **Inference:** a mechanism suggested by source and A/B behavior but not
  isolated by hardware counters.

[`benchmark.csv`](../benchmark.csv) is the candidate ledger. The dated
[`run README`](../runs/rtx4060ti-2026-07-18/README.md) records commands,
five-process medians, oracle gates, residency, and transfers. Failed candidates
remain in both artifacts so later work does not repeat them accidentally.

## Profiling boundary

Nsight Systems captured CUDA API timing but returned no kernel rows on the
recorded driver/tool combination. Nsight Compute 2024.1.1 returned
`ERR_NVGPUCTRPERM`, preserved in
[`cuda_bev_ncu.csv`](../runs/rtx4060ti-2026-07-18/cuda_bev_ncu.csv). Therefore
there is no defensible claim about achieved occupancy, stall breakdown,
coalescing ratio, tensor-pipe utilization, or DRAM saturation.

CUDA builds now include `-lineinfo` by default through `CUDAFLAGS`. After an
administrator enables performance counters, create a fresh directory for each
run and collect both full and source views against a real workload:

```sh
mkdir -p profile/bevfusion-real-frame-baseline/{harness,reports,analysis}
ncu --set full --section PmSampling --section PmSampling_WarpStates \
  -k 'regex:TARGET_KERNEL' -c 1 \
  -o profile/bevfusion-real-frame-baseline/reports/full_frame0 \
  ./build/bevfusion_cuda infer-cuda \
    /data/nuscenes/bevfusion-demo/bevfusion.bfw \
    /data/nuscenes/bevfusion-demo/frame-000.bfi
ncu --set source --section SourceCounters \
  -k 'regex:TARGET_KERNEL' -c 1 \
  -o profile/bevfusion-real-frame-baseline/reports/source_frame0 \
  ./build/bevfusion_cuda infer-cuda \
    /data/nuscenes/bevfusion-demo/bevfusion.bfw \
    /data/nuscenes/bevfusion-demo/frame-000.bfi
```

Do not reuse the existing dated run directory, average different kernels into
one diagnosis, or import Blackwell/Hopper conclusions onto the Ada RTX 4060
Ti. Until counters exist, the highest-confidence optimization evidence remains
the same-workload A/B timer plus oracle and residency gates.

## Next experiments

1. Profile the promoted LSS interval and TransFusion cross-attention kernels
   once counters are enabled; use source-level stall attribution before
   changing memory layout.
2. Revisit branch overlap only on hardware with enough independent resources;
   the current Ada result is a measured rejection, not a universal rule.
3. Reduce the 1.09 GiB full-runtime residency by lifetime aliasing only after a
   graph-level liveness table proves tensors do not overlap.
4. Add official dataset evaluation only when the frame/result contract carries
   the required global metadata.
