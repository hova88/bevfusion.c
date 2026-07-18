# LiDAR branch

The LiDAR path preserves deterministic CPU semantics while keeping the sparse
count on device. Stable grouping produces canonical voxel order, MeanVFE caps
each voxel at ten points, and 21 folded-BN sparse convolutions feed a dense
256-channel BEV.

## Ordering is part of correctness

The CUDA voxelizer first groups by canonical voxel key, then orders groups by
their earliest input point. Point order within each group remains stable. This
matches CPU first-encounter ordering and capacity truncation rather than merely
producing an equivalent unordered set. See
[`src/cuda_voxel.cu`](../src/cuda_voxel.cu).

A real 272,414-point audit exposed an exclusive-scan segment-offset defect.
After repair, all 17,509 voxel coordinates match the CPU path exactly and
MeanVFE maximum absolute error is `1.53e-5`. This is the most useful
correctness lesson in the branch: compact random fixtures did not cover the
segment boundary that the real frame exercised.

## Sparse convolution ladder

| Candidate | Warm time | Decision |
|---|---:|---|
| Direct hash lookup | 89.944 ms | Baseline |
| Shared neighbor per row | 100.587 ms | Reject regression |
| Persistent 27-neighbor rulebook | 70.307 ms | Intermediate |
| 8,192-block grid plus adaptive channel width | about 65.009 ms | Promote backbone |
| Serialized coordinate allocation | 660.398 ms | Reject control |

The complete point-to-BEV stress route measures 58.936 ms warm, including
0.381 ms voxelization, on 100,000 points and 50,000 voxels. Repeated
8,294,400-float outputs are bit exact; the compact real-weight maximum error is
`5.72e-6`. Full occupancies and run samples are in
[`cuda_lidar.txt`](../runs/rtx4060ti-2026-07-18/cuda_lidar.txt).

No NCU counter claim is attached to the rulebook result. **Inference:** reuse
of the precomputed neighborhood is the mechanism suggested by the source and
the A/B ladder; the exact cache, scoreboard, and occupancy contributions
remain unknown until counters can be collected.
