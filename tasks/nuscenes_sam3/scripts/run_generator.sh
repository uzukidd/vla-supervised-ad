#!/usr/bin/env bash
# Local web UI for class prompts, plane choices and scene generation.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m tasks.nuscenes_sam3.python.generator_ui "$@"
