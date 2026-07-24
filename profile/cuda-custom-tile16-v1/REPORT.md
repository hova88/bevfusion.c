# Profiling status: cuda-custom-tile16-v1

No Nsight Systems or Nsight Compute claim is made. CUDA 12.4 compilation is
available, but the environment reports that GPU/NVML access is blocked.
Profiling, hardware-counter interpretation, and bottleneck claims are pending
an accessible RTX 4060 Ti. The first valid measurement must compare the custom
and vendor binaries from the same source revision and workload.

Device access was subsequently restored. No Nsight hardware-counter claim has
yet been made, but end-to-end measurements show the grid-limit-fixed v2 custom
path at `572.971 ms` warm versus the historical `142.381 ms` gate. Correctness
is restored; performance profiling remains future work.
