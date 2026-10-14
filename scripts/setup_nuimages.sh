#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_vla.sh" -m pip install -r "$script_dir/../requirements-nuimages.txt"
