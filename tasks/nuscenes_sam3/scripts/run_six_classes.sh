#!/usr/bin/env bash
# One frame, six cameras, six requested classes; only road/sidewalk become planes.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.six_class "$@"
