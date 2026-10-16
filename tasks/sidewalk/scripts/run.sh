#!/usr/bin/env bash
# One command: original image, dataset ground truth, SAM3 and evaluation.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m shared.python.one_click --task sidewalk "$@"
