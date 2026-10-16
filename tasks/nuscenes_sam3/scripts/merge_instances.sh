#!/usr/bin/env bash
# Reuse saved SAM3 instance points; merge by shared original indices, no model run.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.merge_saved_instances "$@"
