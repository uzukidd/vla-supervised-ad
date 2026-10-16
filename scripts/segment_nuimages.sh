#!/usr/bin/env bash
# Compatibility launcher; implementation lives with its task.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/tasks/nuimages/scripts/segment_nuimages.sh" "$@"
