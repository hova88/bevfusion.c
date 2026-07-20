# Building BEVFusion.c

## Build modes

The CPU and CUDA paths have separate contracts. CPU is always available and
must build without CUDA. CUDA is optional in `AUTO` mode and required only when
the user selects it explicitly.

| Make | CMake | Result |
|---|---|---|
| `make ENABLE_CUDA=0` | `-DBF_ENABLE_CUDA=OFF` | CPU only |
| `make` | `-DBF_ENABLE_CUDA=AUTO` | CPU plus CUDA when complete |
| `make ENABLE_CUDA=1` | `-DBF_ENABLE_CUDA=ON` | fail if CUDA/cuDNN is incomplete |

CPU acceleration is independently optional:

| Make | CMake | Behavior |
|---|---|---|
| `ENABLE_OPENMP=auto` | `BF_ENABLE_OPENMP=AUTO` | use OpenMP when the compiler supports it |
| `ENABLE_BLAS=auto` | `BF_ENABLE_BLAS=AUTO` | use Accelerate on macOS or OpenBLAS when found |
| both set to `0` | both set to `OFF` | dependency-minimal scalar/SIMD fallback |

`AUTO` deliberately does not select a generic Netlib BLAS on Linux: it was
slower than the direct kernels in a real-frame measurement. Install
`libopenblas-dev` if you want Linux BLAS dispatch, or use `ENABLE_BLAS=1` to
explicitly accept another CBLAS implementation. Runtime controls do not require
a rebuild: `BF_CPU_SCALAR=1` selects reference dense kernels and
`BF_CPU_PROFILE=1` prints per-stage latency to stderr. Use `OMP_NUM_THREADS` to
set the CPU thread budget.

Run `make doctor` first. It reports tools, Python modules, CUDA architecture,
cuDNN headers, and all configured asset paths without changing the machine.
The full demo also requires the exact
[`cbgs_bevfusion.pth` checkpoint](https://drive.google.com/file/d/1X50b-8immqlqD8VPAUkSKI0Ls-4k37g9/view?usp=share_link);
download it early and save it under `$NUSCENES_ROOT/checkpoints/` as described
in the [data guide](DATA.md).

## Ubuntu, Debian, and WSL2

For the portable CPU build:

```sh
sudo apt update
sudo apt install build-essential cmake python3 python3-venv
make ENABLE_CUDA=0 test
```

Optional optimized CPU packages:

```sh
sudo apt install libopenblas-dev libomp-dev
```

GCC normally supplies its OpenMP runtime with the compiler; Clang commonly
uses `libomp-dev`. Neither package is required for the scalar fallback.

WSL2 CUDA uses the Windows NVIDIA driver exposed to Linux. Do not install a
second Linux display driver inside WSL. Install a compatible CUDA toolkit and
cuDNN development package, confirm `nvcc`, `nvidia-smi`, and `cudnn.h` in
`make doctor`, then run `make ENABLE_CUDA=1`.

## macOS

The supported path is CPU inference and all dependency-free tests.
Apple does not provide a current NVIDIA CUDA stack for modern macOS.

```sh
xcode-select --install
brew install cmake              # only if CMake is desired
make ENABLE_CUDA=0 test
```

The build uses C11, `mmap`, `clock_gettime`, the system math library, and the
built-in Accelerate framework. OpenMP is used only when a compatible compiler
and runtime are present. Real-frame CPU inference is not described as real
time until it is measured on the target Mac.

## NVIDIA Jetson

Use the CUDA and cuDNN versions bundled for the installed JetPack release;
mixing desktop repository packages with JetPack is a common ABI failure. Begin
with a native build on the device:

```sh
make doctor
make ENABLE_CUDA=1 CUDA_ARCH=sm_87 -j2  # Jetson Orin
```

Common architectures are `sm_87` for Orin, `sm_72` for Xavier, and `sm_53`
for Nano. Verify the exact module. Older Jetson toolkits and this source may
also differ in supported C++/cuDNN APIs; explicit CUDA mode is designed to
surface that at configure time.

The measured production runtime owns about 1.1 GiB of device memory, before
driver/runtime reserve. That fits some Jetson configurations but not all
concurrent workloads. Reduce competing GPU processes and measure the target;
there is not yet a configurable low-memory CUDA plan.

## CUDA discovery overrides

Make accepts normal compiler variables:

```sh
make ENABLE_CUDA=1 \
  NVCC=/opt/cuda/bin/nvcc \
  CPPFLAGS='-Iinclude -I/opt/cudnn/include' \
  CUDAFLAGS='-O3 -lineinfo -arch=sm_90 -I/opt/cudnn/include' \
  LDFLAGS='-L/opt/cudnn/lib64'
```

For CMake:

```sh
cmake -S . -B build/custom \
  -DBF_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/opt/cuda \
  -DCUDNN_ROOT=/opt/cudnn \
  -DBF_CUDA_ARCHS=90
cmake --build build/custom -j
```

## Troubleshooting

`nvcc: command not found` means only the CUDA compiler is missing; use the CPU
build immediately or set `NVCC`. `cudnn.h: No such file` means the runtime
package alone is insufficient—development headers are required. An
`unsupported gpu architecture` error means `CUDA_ARCH` does not match that
toolkit. A runtime `no kernel image` error means the binary omitted the target
GPU architecture.

For reproducible release artifacts, leave `NATIVE=0`. Enable `NATIVE=1` only
for a binary that stays on the machine where it was built.

For a memory-safety development build:

```sh
cmake --preset dev
cmake --build --preset dev -j2
ctest --preset dev
```
