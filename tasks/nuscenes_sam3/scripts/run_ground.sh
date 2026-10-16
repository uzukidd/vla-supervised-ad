#!/usr/bin/env bash
# Reuse saved six-camera masks/points; merge road and sidewalk regions into planes.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.ground_regions "$@"
