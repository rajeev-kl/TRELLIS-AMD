#!/usr/bin/env bash
# Create the Python environment for TRELLIS-AMD on this machine.
#
#   ./setup_venv.sh          # create .venv and install everything
#   ./setup_venv.sh --force  # delete an existing .venv first
#
# Replaces steps 1-4 and 8 of the repo's install_amd.sh, which is not usable
# here (it installs ROCm 6.4 wheels with no gfx1151 kernels). Run order for a
# clean machine:
#
#   sudo ./setup_root.sh        # headers (only privileged step)
#   ./prepare_rocm_headers.sh   # vendor rocPRIM / rocThrust / hipCUB
#   ./setup_venv.sh             # this script
#   ./build_extensions.sh       # the three HIP extensions
#   ./run_app.sh                # http://127.0.0.1:7860
#
# See SETUP_NOTES.md for why each pin and patch exists.

set -euo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# torch 2.10.0 is the version the fork was tested with; rocm7.1 is the wheel
# index that actually ships gfx1151 kernels (6.4 and 7.0 do not, or are older
# than this machine's runtime). Verified: gfx1151 appears in
# torch.cuda.get_arch_list().
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
ROCM_INDEX="https://download.pytorch.org/whl/rocm7.1"

if [[ "${1:-}" == "--force" ]] && [[ -d .venv ]]; then
    echo "[0/5] Removing existing .venv..."
    rm -rf .venv
fi

echo "=============================================="
echo "  TRELLIS-AMD Python environment"
echo "=============================================="
echo

echo "[1/5] Creating virtualenv..."
if [[ -d .venv ]]; then
    echo "      .venv already exists — reusing (use --force to recreate)"
else
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -c "import sys; assert sys.prefix != sys.base_prefix, 'venv not active'"
echo "      python: $(python -V)  at $(command -v python)"
echo

echo "[2/5] Upgrading pip tooling..."
pip install --quiet --upgrade pip wheel setuptools
echo

echo "[3/5] Installing PyTorch ${TORCH_VERSION} for ROCm..."
pip install --quiet "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${ROCM_INDEX}"
python - <<'PY'
import torch
archs = torch.cuda.get_arch_list()
print(f"      torch {torch.__version__}  hip {torch.version.hip}")
print(f"      gfx1151 in arch list: {'gfx1151' in archs}")
assert 'gfx1151' in archs, "this torch build has no gfx1151 kernels — wrong ROCm index?"
PY
echo

echo "[4/5] Installing TRELLIS dependencies..."
# requirements.txt carries the pins that keep gradio 4.44.1 working against
# modern resolvers (transformers / huggingface_hub / fastapi). See SETUP_NOTES.md.
pip install --quiet -r requirements.txt
echo

echo "[5/5] Patching gradio_client for boolean JSON schemas..."
# gradio_litmodel3d produces `additionalProperties: true`, which gradio_client's
# schema walker assumes is always a dict. Same fix as step 8 of install_amd.sh,
# but idempotent and it fails loudly instead of silently no-oping like the
# upstream sed does.
python - <<'PY'
import pathlib, sysconfig
p = pathlib.Path(sysconfig.get_paths()["purelib"]) / "gradio_client" / "utils.py"
src = p.read_text()
GUARD = ('    # Handle non-dict schemas (e.g. boolean from additionalProperties: true)\n'
         '    if not isinstance(schema, dict):\n        return "Any"\n')
SIGS = ('def get_type(schema: dict):\n',
        'def _json_schema_to_python_type(schema: Any, defs) -> str:\n')
if GUARD.strip() in src:
    print("      already patched — skipping")
else:
    backup = p.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(src)
    for sig in SIGS:
        if sig not in src:
            raise SystemExit(f"      FAILED: signature not found, gradio_client changed: {sig!r}")
        src = src.replace(sig, sig + GUARD, 1)
    p.write_text(src)
    print(f"      patched {p.name} (backup at {backup.name})")
PY
echo

echo "=============================================="
echo "  Environment ready."
echo "  Next:  ./build_extensions.sh"
echo "=============================================="
