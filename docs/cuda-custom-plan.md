# Custom CUDA KDA plan

1. Preserve the cuDNN/cuBLAS implementation behind `BF_CUDA_VENDOR` and build
   it as a separately named executable.
2. Compile the default provider from CUDA Runtime, CUB headers, and the internal
   strict-FP32 convolution/GEMM boundary only.
3. Run focused operator fixtures, then Swin, camera, BEV, TransFusion, CUDA tail,
   LiDAR/LSS, and full-runtime oracles in that order.
4. Measure each passing candidate on the same RTX 4060 Ti frame with five fresh
   processes for cold and warm medians. Record VRAM and boundary transfers.
5. Profile only after correctness. Tune separately for Swin patch/projections,
   batch-six camera neck, wide 180x180 BEV convolutions, and small TransFusion
   projections. Retain every rejected candidate in the ledger.
6. Promote only when the numerical, latency, memory, transfer, repeatability,
   Ada, and Turing gates in `cuda-custom-draft.md` all pass.
