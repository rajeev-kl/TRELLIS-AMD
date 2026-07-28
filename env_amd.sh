# Environment for TRELLIS on AMD Strix Halo / gfx1151 — source this, don't execute it.
#
#   source env_amd.sh
#
# Nothing machine-specific is hardcoded here. Everything is either auto-detected
# or read from a local `.env` (gitignored). Copy `.env.example` to `.env` to
# override anything below.

_TRELLIS_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------------------------------------------------------------------------
# Local overrides from .env (KEY=VALUE, # comments allowed). Never committed.
# ---------------------------------------------------------------------------
if [ -f "${_TRELLIS_ROOT}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "${_TRELLIS_ROOT}/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# CRITICAL: swap in the system HSA runtime.
#
# The ROCm HSA runtime bundled inside the PyTorch wheel SIGSEGVs on gfx1151
# during HSA queue creation, before any TRELLIS code runs. Even
# `torch.arange(10, device='cuda')` crashes:
#
#   rocr::AMD::GpuAgent::ReleaseQueueMainScratch(ScratchCache::ScratchInfo&)
#   rocr::AMD::GpuAgent::QueueCreate(...) [clone .cold]      <-- failure path
#   rocr::AMD::GpuAgent::CreateInterceptibleQueue(...)
#   rocr::AMD::GpuAgent::InitDma()
#
# QueueCreate fails on this APU and its error-cleanup branch dereferences a null
# scratch pointer. The system ROCm runtime does not have the bug, so preload it
# over the bundled one. Without this, nothing GPU-related works at all.
#
# Override with TRELLIS_HSA_RUNTIME in .env if your ROCm lives elsewhere.
# ---------------------------------------------------------------------------
if [ -z "${TRELLIS_HSA_RUNTIME:-}" ]; then
    for _c in "${ROCM_PATH:-/opt/rocm}/lib/libhsa-runtime64.so.1" \
              /opt/rocm/lib/libhsa-runtime64.so.1 \
              /opt/rocm/core-*/lib/libhsa-runtime64.so.1
    do
        [ -e "${_c}" ] && { TRELLIS_HSA_RUNTIME="${_c}"; break; }
    done
    unset _c
fi

if [ -n "${TRELLIS_HSA_RUNTIME:-}" ] && [ -e "${TRELLIS_HSA_RUNTIME}" ]; then
    export LD_PRELOAD="${TRELLIS_HSA_RUNTIME}${LD_PRELOAD:+:${LD_PRELOAD}}"
else
    echo "env_amd.sh WARNING: system HSA runtime not found." >&2
    echo "  Searched \$ROCM_PATH/lib and /opt/rocm{,/core-*}/lib for" >&2
    echo "  libhsa-runtime64.so.1. Locate it with:" >&2
    echo "    find /opt/rocm -name 'libhsa-runtime64.so.1'" >&2
    echo "  then set TRELLIS_HSA_RUNTIME=<path> in .env" >&2
    echo "  Without this preload, all GPU ops segfault on gfx1151." >&2
fi

# ---------------------------------------------------------------------------
# ROCm location. This install uses a component layout (/opt/rocm/include is a
# symlink into a versioned dir), so resolve it rather than hardcode a version.
# ---------------------------------------------------------------------------
if [ -z "${ROCM_HOME:-}" ]; then
    if [ -e /opt/rocm/include ]; then
        ROCM_HOME="$( dirname "$( readlink -f /opt/rocm/include )" )"
    else
        ROCM_HOME=/opt/rocm
    fi
fi
export ROCM_HOME
export ROCM_PATH="${ROCM_PATH:-${ROCM_HOME}}"
export HIP_PLATFORM="${HIP_PLATFORM:-amd}"

# ---------------------------------------------------------------------------
# Target architecture. Auto-detected; this fork exists for gfx1151.
# ---------------------------------------------------------------------------
if [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
    PYTORCH_ROCM_ARCH="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' \
                         | grep -v generic | head -1)"
    [ -z "${PYTORCH_ROCM_ARCH}" ] && PYTORCH_ROCM_ARCH=gfx1151
fi
export PYTORCH_ROCM_ARCH
case "${PYTORCH_ROCM_ARCH}" in
    gfx1151) : ;;
    *) echo "env_amd.sh NOTE: detected ${PYTORCH_ROCM_ARCH}, but this fork is" \
            "validated only on gfx1151 (Strix Halo)." >&2 ;;
esac

# ---------------------------------------------------------------------------
# Model weight caches (optional).
#
# No weights live in this repo; they are fetched on first use. There are THREE
# independent caches, each with its own env var — HF_HOME covers only the first:
#
#   HF_HOME     huggingface_hub -> TRELLIS checkpoints, CLIP text encoder  ~9.2 GB
#   TORCH_HOME  torch.hub       -> DINOv2 image encoder                    ~1.2 GB
#                                  (trellis_image_to_3d.py)
#   U2NET_HOME  rembg           -> u2net.onnx background remover           ~168 MB
#                                  (preprocess_image() uses it on every run
#                                   unless the input already has alpha;
#                                   rembg/sessions/base.py reads U2NET_HOME
#                                   and defaults to ~/.u2net)
#
# Set the TRELLIS_* variants in .env to keep ~10.5 GB off your root filesystem.
# Left unset, each library's own default (~/.cache, ~/.u2net) applies.
# Deleting any of them is never data loss — only a re-download.
# ---------------------------------------------------------------------------
[ -n "${TRELLIS_HF_HOME:-}" ]    && export HF_HOME="${TRELLIS_HF_HOME}"
[ -n "${TRELLIS_TORCH_HOME:-}" ] && export TORCH_HOME="${TRELLIS_TORCH_HOME}"
[ -n "${TRELLIS_U2NET_HOME:-}" ] && export U2NET_HOME="${TRELLIS_U2NET_HOME}"

# ---------------------------------------------------------------------------
# Make `import trellis` work regardless of where the invoked script lives.
# Python puts the *script's* directory on sys.path, not the cwd.
# ---------------------------------------------------------------------------
export PYTHONPATH="${_TRELLIS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# TRELLIS backend selection
# ---------------------------------------------------------------------------
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"          # xformers is CUDA-only
export XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}"
export SPARSE_BACKEND="${SPARSE_BACKEND:-torchsparse}"  # spconv is CUDA-only
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

# Required for PyTorch's AOTriton flash / mem-efficient SDPA kernels on RDNA.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"

# ---------------------------------------------------------------------------
# Build-time include path for the vendored header-only ROCm libs
# (rocPRIM / rocThrust / hipCUB — see prepare_rocm_headers.sh). Harmless at runtime.
# ---------------------------------------------------------------------------
export CPLUS_INCLUDE_PATH="${_TRELLIS_ROOT}/.rocm-headers/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

unset _TRELLIS_ROOT
