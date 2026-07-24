#!/bin/sh
# Print an nvcc -arch value for the local GPU. An empty result intentionally
# leaves architecture selection to CMake/nvcc or to the user's CUDA_ARCH.
set -eu

if command -v nvidia-smi >/dev/null 2>&1; then
    raw_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) ||
        raw_capability=
    capability=$(printf '%s\n' "$raw_capability" |
        sed -n '1{s/[.]//g;s/^[[:space:]]*//;s/[[:space:]]*$//;p;}')
    if [ -n "$capability" ]; then
        printf 'sm_%s\n' "$capability"
        exit 0
    fi
fi

model=$(tr -d '\000' </proc/device-tree/model 2>/dev/null || true)
case "$model" in
    *Orin*) printf 'sm_87\n' ;;
    *Xavier*) printf 'sm_72\n' ;;
    *Nano*) printf 'sm_53\n' ;;
esac
