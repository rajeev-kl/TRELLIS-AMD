#!/usr/bin/env bash
# TRELLIS-AMD — privileged setup steps
#
# This is the ONLY part of the TRELLIS-AMD install that needs root.
# Everything else (venv, PyTorch, HIP extension builds) runs unprivileged.
#
# Usage:
#   sudo ./setup_root.sh
#
# What it does:
#   1. Installs the build-time *header* packages the three HIP extensions need.
#      See the annotated PKGS list below for what each one is for and which
#      #include it satisfies. All of the ROCm ones come from the already-configured
#      repo.radeon.com/rocmradeon/apt/26.13 repo, pinned to 7.14 so they match the
#      installed amdrocm-runtime7.14 exactly.
#   2. Verifies every expected header actually landed.
#   3. Verifies this user is in the 'video' and 'render' groups (ROCm GPU access).
#
# Nothing here upgrades or replaces ROCm, touches the GPU driver, or changes kernel
# parameters. ROCm 7.14 is already installed and working on this machine; this only
# adds the matching headers for the version that is already present. The runtime
# .so files these headers describe are all present already — PyTorch links against
# its own bundled copies.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "ERROR: this script must run as root:  sudo ./setup_root.sh" >&2
    exit 1
fi

# The unprivileged user who owns the checkout (not root).
TARGET_USER="${SUDO_USER:-$(stat -c '%U' "$(dirname "$(readlink -f "$0")")")}"

echo "=============================================="
echo "  TRELLIS-AMD — privileged setup"
echo "=============================================="
echo

export DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Package list
#
# The installed ROCm is runtime-only: it ships the .so files but almost none of
# the headers. PyTorch's own HIP headers — which every extension reaches via
# torch/extension.h -> ATen/hip/HIPContext.h — pull in a chain of ROCm library
# headers, so that whole chain must be present even though the extensions never
# call those libraries directly.
#
# This list is not guesswork. It was derived by enumerating every ROCm header
# referenced under torch/include, then verified by compiling a probe translation
# unit (torch/extension.h + ATen/hip/HIPContext.h + hipcub + thrust) against the
# extracted .debs until it compiled clean.
# ---------------------------------------------------------------------------
# The 'ROCm for Radeon' packages are version-suffixed (e.g. amdrocm-blas-dev7.14).
# Derive the suffix from the already-installed runtime so this keeps working after
# a ROCm upgrade, instead of hardcoding a version.
ROCM_PKG_SUFFIX="${ROCM_PKG_SUFFIX:-$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
    | grep -oP '^amdrocm-runtime\K[0-9]+\.[0-9]+$' | sort -V | tail -1)}"
if [[ -z "${ROCM_PKG_SUFFIX}" ]]; then
    echo "ERROR: could not detect the installed amdrocm runtime version." >&2
    echo "  Expected a package like 'amdrocm-runtime7.14'. Check with:" >&2
    echo "    dpkg -l | grep amdrocm-runtime" >&2
    echo "  Then re-run with an explicit suffix, e.g.:" >&2
    echo "    sudo ROCM_PKG_SUFFIX=7.14 ./setup_root.sh" >&2
    exit 1
fi
echo "Detected ROCm package suffix: ${ROCM_PKG_SUFFIX}"

# Python dev headers must match the interpreter that will build the extensions.
PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

S="${ROCM_PKG_SUFFIX}"
PKGS=(
    libsparsehash-dev            # <google/dense_hash_map>  (torchsparse CPU hashmap)
    "python${PY_MM}-dev"         # Python.h                 (all extensions)
    "amdrocm-runtime-dev${S}"    # hip/hip_runtime.h         (all extensions)
    "amdrocm-blas-dev${S}"       # hipblas/hipblas.h         (torch hdrs; hipified <cublas_v2.h>)
    "amdrocm-hipblas-common-dev${S}"  # hipblas-common/...   (included BY hipblas.h)
    "amdrocm-sparse-dev${S}"     # hipsparse/hipsparse.h     (ATen/hip/HIPContextLight.h)
    "amdrocm-solver-dev${S}"     # hipsolver/hipsolver.h     (torch headers)
    "amdrocm-fft-dev${S}"        # hipfft/hipfft.h           (torch headers)
    "amdrocm-rand-dev${S}"       # hiprand/hiprand_kernel.h  (torch headers)
    "amdrocm-dnn-dev${S}"        # miopen/miopen.h           (torch headers)
    "amdrocm-rccl-dev${S}"       # rccl/rccl.h               (torch headers)
)

NEED=()
for pkg in "${PKGS[@]}"; do
    dpkg -s "${pkg}" >/dev/null 2>&1 || NEED+=("${pkg}")
done

echo "[1/3] Installing build dependencies..."
if [[ ${#NEED[@]} -eq 0 ]]; then
    echo "      all already installed — skipping"
else
    echo "      to install: ${NEED[*]}"
    apt-get update
    apt-get install -y --no-install-recommends "${NEED[@]}"
fi
echo

echo "[2/3] Verifying headers landed..."
fail=0
check_hdr() {
    if [[ -e "$1" ]]; then
        echo "      OK:      $1"
    else
        echo "      MISSING: $1" >&2
        fail=1
    fi
}
check_hdr /usr/include/sparsehash/dense_hash_map
check_hdr "/usr/include/python${PY_MM}/Python.h"

# /opt/rocm/include is a symlink into a versioned component dir — resolve it
# rather than hardcoding the version.
ROCM_INC="$( [[ -e /opt/rocm/include ]] && readlink -f /opt/rocm/include || echo /opt/rocm/include )"
echo "      (ROCm include dir: ${ROCM_INC})"
for h in hip/hip_runtime.h hipblas/hipblas.h hipblas-common/hipblas-common.h \
         hipsparse/hipsparse.h hipsolver/hipsolver.h hipfft/hipfft.h \
         hiprand/hiprand_kernel.h miopen/miopen.h rccl/rccl.h
do
    check_hdr "${ROCM_INC}/${h}"
done
if [[ ${fail} -ne 0 ]]; then
    echo >&2
    echo "ERROR: expected headers are still missing — see above." >&2
    exit 1
fi
echo

echo "[3/3] Checking GPU group membership for '${TARGET_USER}'..."
missing_groups=()
for grp in video render; do
    if id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx "${grp}"; then
        echo "      OK: ${TARGET_USER} is in '${grp}'"
    else
        missing_groups+=("${grp}")
    fi
done

if [[ ${#missing_groups[@]} -gt 0 ]]; then
    for grp in "${missing_groups[@]}"; do
        echo "      adding ${TARGET_USER} to '${grp}'"
        usermod -aG "${grp}" "${TARGET_USER}"
    done
    echo
    echo "      NOTE: log out and back in for the new group membership to apply."
fi
echo

echo "=============================================="
echo "  Privileged setup complete."
echo "  Return to Claude Code — the rest of the"
echo "  install needs no root."
echo "=============================================="
