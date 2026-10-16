#!/usr/bin/env bash
# One command: EFFOcc inference, six-camera projection, SAM3 and final comparison.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m shared.python.one_click --task effocc "$@"
