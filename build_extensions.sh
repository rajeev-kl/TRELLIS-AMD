#!/usr/bin/env bash
# Build the three HIP extensions for gfx1151 (Strix Halo).
#
#   ./build_extensions.sh            # build all three
#   ./build_extensions.sh torchsparse  # build just one
#
# This replaces steps 5-7 of the repo's install_amd.sh. That script is not
# usable as-is on this machine:
#   * it installs torch from the ROCm 6.4 index (no gfx1151 kernels)
#   * it hardcodes PYTORCH_ROCM_ARCH=gfx1100 for torchsparse (wrong arch here)
# Steps 1-4 and 8 (venv, torch, requirements, gradio_client patch) are already
# done. See env_amd.sh for the machine-specific environment.
#
# Prerequisite: sudo ./setup_root.sh   (installs HIP dev headers + sparsehash)

set -euo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source env_amd.sh

WHICH="${1:-all}"

# --- preflight -------------------------------------------------------------
echo "=============================================="
echo "  Building TRELLIS-AMD extensions"
echo "  arch: ${PYTORCH_ROCM_ARCH}   rocm: ${ROCM_HOME}"
echo "=============================================="
echo

missing=0
for hdr in \
    "${ROCM_HOME}/include/hip/hip_runtime.h" \
    "${ROCM_HOME}/include/hipblas/hipblas.h" \
    "$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')/Python.h" \
    /usr/include/sparsehash/dense_hash_map \
    .rocm-headers/include/hipcub/hipcub.hpp \
    .rocm-headers/include/thrust/scan.h \
    .rocm-headers/include/rocprim/rocprim.hpp
do
    if [[ ! -e "${hdr}" ]]; then
        echo "MISSING: ${hdr}" >&2
        missing=1
    fi
done
if [[ ${missing} -ne 0 ]]; then
    echo >&2
    echo "Required headers are missing. Run:  sudo ./setup_root.sh" >&2
    exit 1
fi
echo "Preflight: all required headers present."
echo

DETECTED_ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | grep -v 'generic' | head -1)"
if [[ "${DETECTED_ARCH}" != "${PYTORCH_ROCM_ARCH}" ]]; then
    echo "WARNING: rocminfo reports '${DETECTED_ARCH}' but PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}" >&2
fi

build_nvdiffrast() {
    echo ">>> [1/3] nvdiffrast-hip"
    # Uses torch.utils.cpp_extension, which picks up ROCM_HOME and hipcc itself.
    ( cd extensions/nvdiffrast-hip && pip install . --no-build-isolation )
    echo
}

build_dgr() {
    echo ">>> [2/3] diff-gaussian-rasterization (manual HIP build)"
    # build_hip.sh auto-detects the arch from rocminfo, so it gets gfx1151
    # right with no patch needed. It requires an active venv (checks
    # $VIRTUAL_ENV) and picks up thrust/hipcub via CPLUS_INCLUDE_PATH.
    ( cd extensions/diff-gaussian-rasterization && chmod +x build_hip.sh && ./build_hip.sh )
    echo
}

build_torchsparse() {
    echo ">>> [3/3] torchsparse"
    # FORCE_CUDA=1 is what makes it build the HIP/GPU backend rather than
    # falling back to CPU-only.
    ( cd extensions/torchsparse \
        && rm -rf build ./*.egg-info \
        && FORCE_CUDA=1 pip install . --no-build-isolation )
    echo
}

case "${WHICH}" in
    all)          build_nvdiffrast; build_dgr; build_torchsparse ;;
    nvdiffrast)   build_nvdiffrast ;;
    dgr|diff-gaussian-rasterization) build_dgr ;;
    torchsparse)  build_torchsparse ;;
    *) echo "unknown target: ${WHICH}" >&2; exit 1 ;;
esac

echo "=============================================="
echo "  Verifying imports"
echo "=============================================="
python - <<'PY'
import importlib, traceback
ok = True
for mod, attr in [("nvdiffrast.torch", None),
                  ("diff_gaussian_rasterization", None),
                  ("torchsparse", "__version__")]:
    try:
        m = importlib.import_module(mod)
        extra = f"  ({getattr(m, attr, '?')})" if attr else ""
        print(f"  OK      {mod}{extra}")
    except Exception:
        ok = False
        print(f"  FAIL    {mod}")
        traceback.print_exc()

# torchsparse must have its compiled GPU backend, not the CPU fallback
try:
    import torchsparse.backend as b
    has = hasattr(b, "hash_forward") or hasattr(b, "hash_cuda")
    print(f"  torchsparse backend loaded, gpu symbols present: {has}")
except Exception as e:
    ok = False
    print(f"  FAIL    torchsparse.backend -> {type(e).__name__}: {e}")

raise SystemExit(0 if ok else 1)
PY

echo
echo "Build complete. Run the app with:  ./run_app.sh"
