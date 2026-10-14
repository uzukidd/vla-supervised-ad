#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_vla.sh" "$script_dir/../python/sam3_sidewalk.py" --mode custom "$@"
