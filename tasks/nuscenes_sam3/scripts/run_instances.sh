#!/usr/bin/env bash
# One frame, all six cameras: SAM3 instances -> LiDAR XY points -> convex hulls.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.instance_bev "$@"
