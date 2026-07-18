# Execution graph

Two sensor branches produce metric-aligned BEV features. Their concatenation
feeds a shared BEV backbone and a TransFusion decoder.

![Parallel sensor branches joining at BEV fusion](assets/pipeline.svg)

| Segment | Main tensor transition | CUDA owner |
|---|---|---|
| Depth raster | points and calibration to `[6,1,256,704]` | [`src/cuda_depth_raster.cu`](../src/cuda_depth_raster.cu) |
| Camera encoder | images to three Swin feature levels | [`src/cuda_swin.cu`](../src/cuda_swin.cu) |
| Camera neck | feature levels plus depth to logits/context | [`src/cuda_camera.cu`](../src/cuda_camera.cu) |
| Lift-splat | context to `[1,80,360,360]` | [`src/cuda_lss.cu`](../src/cuda_lss.cu) |
| Image downsample | image BEV to `[1,80,180,180]` | [`src/cuda_camera.cu`](../src/cuda_camera.cu) |
| Voxel and VFE | points to bounded sparse features | [`src/cuda_voxel.cu`](../src/cuda_voxel.cu) |
| Sparse backbone | sparse features to `[1,256,180,180]` | [`src/cuda_lidar.cu`](../src/cuda_lidar.cu) |
| Fusion | concatenate to `[1,336,180,180]` | [`src/cuda_runtime.cu`](../src/cuda_runtime.cu) |
| Detection tail | BEV stage, decoder, metric boxes | [`src/cuda_bev_stage.cu`](../src/cuda_bev_stage.cu), [`src/cuda_transfusion.cu`](../src/cuda_transfusion.cu) |

The promoted runtime uses one stream. A two-stream camera/LiDAR schedule is
implemented as an explicit control. **Measured:** on the recorded RTX 4060 Ti
frame its warm median was 145.783 ms, versus 142.381 ms serial, so resource
contention made overlap a 2.4% regression and the candidate was rejected.

The graph ends with a single device-to-host copy of the 8,804-byte canonical
detection structure. That synchronization is deliberate: the public API
returns a host value, not a future.
