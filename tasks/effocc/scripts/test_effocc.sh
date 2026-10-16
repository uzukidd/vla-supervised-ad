#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/shared/scripts/run_vla.sh" -m unittest discover -s tasks/effocc/tests -t . -v "$@"
