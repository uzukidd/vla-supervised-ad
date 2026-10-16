#!/usr/bin/env bash
# New independent camera-only task: SAM3 masks projected onto global z=0 ground.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.flat_ground.python.project_semantics "$@"
