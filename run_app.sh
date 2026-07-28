#!/usr/bin/env bash
# Launch TRELLIS-AMD on this machine.
#
#   ./run_app.sh              # image-to-3D Gradio UI  -> http://localhost:7860
#   ./run_app.sh app_text.py  # text-to-3D UI
#   ./run_app.sh example.py   # headless staged example
#
# env_amd.sh carries the machine-specific settings, most importantly the
# LD_PRELOAD of the system HSA runtime — without it every GPU op segfaults on
# gfx1151. See the comments in env_amd.sh for the backtrace and reasoning.
#
# Heads-up from the repo README: GLB export takes 5-10 minutes and pushes both
# CPU and GPU hard. That is expected, not a hang. Console shows steps 1/5..5/5.

set -euo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source env_amd.sh

exec python "${@:-app.py}"
