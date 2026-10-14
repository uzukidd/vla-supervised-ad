#!/usr/bin/env bash
# One LiDAR keyframe, all six cameras, semantic point-cloud export by default.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_vla.sh" "$script_dir/../python/sam3_nuscenes.py" \
  --mode lidarseg \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT \
  --max-images 6 --export-pointcloud "$@"
