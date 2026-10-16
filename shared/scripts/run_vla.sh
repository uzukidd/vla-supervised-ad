#!/usr/bin/env bash
# Run Python exclusively in WSL's vla-supervised Conda environment.
set -euo pipefail
if [[ "$(uname -s)" != Linux ]] || ! grep -qi microsoft /proc/sys/kernel/osrelease; then
    printf 'This entry point requires WSL.\n' >&2
    exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" == vla-supervised ]]; then
    exec python "$@"
fi
conda_executable="${CONDA_EXE:-}"
if [[ ! -x "$conda_executable" ]]; then
    conda_executable=""
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            conda_executable="$candidate"
            break
        fi
    done
fi
if [[ -z "$conda_executable" ]]; then
    conda_executable="$(command -v conda || true)"
fi
if [[ ! -x "$conda_executable" ]]; then
    printf 'Conda was not found. Activate vla-supervised or set CONDA_EXE.\n' >&2
    exit 1
fi
exec "$conda_executable" run --no-capture-output -n vla-supervised python "$@"
