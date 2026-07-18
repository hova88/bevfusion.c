# Executable plan

1. **Container and contract** — export a deterministic, aligned, checksummed
   BFW file; mmap and hash-index it in C; reject malformed files.
2. **Oracle harness** — load the original OpenPCDet model and save deterministic
   stage fixtures plus calibration-aware real-frame inputs.
3. **Scalar CPU graph** — implement preprocessing and every operator with arena
   lifetime planning; compare stage-by-stage with the oracle.
4. **Optimized CPU graph** — add sparse-coordinate specialization, tiling,
   OpenMP and SIMD behind reference switches; measure full frames.
5. **Strict CUDA graph** — persistent weights and activations, cuDNN where it is
   the measured winner, custom sparse/LSS/top-k/decode kernels, narrow C ABI.
6. **KDA kernel loop** — profile hot kernels, maintain candidates and fallback
   switches, validate numerically after each promoted change.
7. **Canonical decode and BEV TUI** — retain validated BFI mappings for the
   viewer lifetime; render density/height-aware LiDAR occupancy under
   confidence-filled metric boxes, velocity, rings, trails, and inspection;
   preserve robust terminal restoration and a deterministic compose fixture.
8. **Completion gates** — sanitizers, malformed inputs, CPU/CUDA oracles, real
   samples, official evaluation, cold/warm benchmarks, and peak memory reports.
