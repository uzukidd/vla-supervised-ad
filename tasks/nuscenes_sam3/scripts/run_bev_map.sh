#!/usr/bin/env bash
# Add metric BEV Map mode to an exported scene without rerunning segmentation.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.scene_bev_map "$@"
