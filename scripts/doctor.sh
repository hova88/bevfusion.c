#!/bin/sh
set -u

CC=${CC:-cc}
NVCC=${NVCC:-nvcc}
PYTHON=${PYTHON:-python3}

found() { command -v "$1" 2>/dev/null || printf 'not found'; }
python_module() {
    "$PYTHON" - "$1" <<'PY' 2>/dev/null
import importlib.util
import sys
print("yes" if importlib.util.find_spec(sys.argv[1]) else "no")
PY
}

printf 'BEVFusion.c environment\n'
printf '  host:         %s / %s\n' "$(uname -s 2>/dev/null || printf unknown)" "$(uname -m 2>/dev/null || printf unknown)"
printf '  C compiler:   %s\n' "$(found "$CC")"
printf '  CMake:        %s\n' "$(found cmake)"
printf '  Python:       %s\n' "$(found "$PYTHON")"
printf '  CPU kernels:  BLAS=%s OpenMP=%s (BF_CPU_SCALAR=1 forces oracle)\n' \
    "${BF_BLAS:-auto}" "${BF_OPENMP:-auto}"

if command -v "$PYTHON" >/dev/null 2>&1; then
    printf '  Python tools: numpy=%s Pillow=%s nuscenes=%s torch=%s\n' \
        "$(python_module numpy)" "$(python_module PIL)" \
        "$(python_module nuscenes)" "$(python_module torch)"
fi

if command -v "$NVCC" >/dev/null 2>&1; then
    cuda_version=$($NVCC --version 2>/dev/null | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n 1)
    cuda_arch=$("$(dirname "$0")/detect_cuda_arch.sh" 2>/dev/null || true)
    printf '  CUDA:         %s (version %s, arch %s)\n' "$(found "$NVCC")" \
        "${cuda_version:-unknown}" "${cuda_arch:-set CUDA_ARCH manually}"
    if printf '#include <cudnn.h>\n' | "$CC" -E -x c - >/dev/null 2>&1; then
        printf '  cuDNN headers: yes\n'
    else
        printf '  cuDNN headers: no (CPU still builds; CUDA needs development headers)\n'
    fi
else
    printf '  CUDA:         optional, not found (CPU build is available)\n'
fi

printf '  nuScenes:     %s\n' "${NUSCENES_ROOT:-not configured}"
for item in CHECKPOINT MODEL DEMO_MANIFEST; do
    case "$item" in
        CHECKPOINT) path=${CHECKPOINT:-} ;;
        MODEL) path=${MODEL:-} ;;
        DEMO_MANIFEST) path=${DEMO_MANIFEST:-} ;;
    esac
    label=$(printf '%s' "$item" | tr '[:upper:]_' '[:lower:] ')
    if [ -n "$path" ] && [ -f "$path" ]; then
        printf '  %-13s %s\n' "$label:" "$path"
    else
        printf '  %-13s missing%s\n' "$label:" "${path:+ ($path)}"
    fi
done

printf '\nNext safe step: make test (no dataset, checkpoint, Python packages, or CUDA required).\n'
