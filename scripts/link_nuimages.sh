#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source=/mnt/d/datasets/nuimages
destination="$project_root/datasets/nuimages"
if [[ ! -d "$source" ]]; then
    printf 'Source directory does not exist: %s\n' "$source" >&2
    exit 1
fi
if [[ -L "$destination" ]]; then
    if [[ "$(readlink -f -- "$destination")" == "$(readlink -f -- "$source")" ]]; then
        printf 'Already linked: %s -> %s\n' "$destination" "$source"
        exit 0
    fi
    printf 'Destination has a different symlink: %s\n' "$destination" >&2
    exit 1
fi
if [[ -e "$destination" ]]; then
    printf 'Destination already exists: %s\n' "$destination" >&2
    exit 1
fi
mkdir -p -- "$project_root/datasets"
ln -s -- "$source" "$destination"
printf 'Linked: %s -> %s\n' "$destination" "$source"
