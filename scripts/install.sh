#!/usr/bin/env bash
# NOTE: no `-u`. conda's gcc_linux-64 deactivate hook dereferences
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED, which aborts the install
# under `set -u` on a clean environment.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "TRACE requires Linux with an NVIDIA GPU." >&2
  exit 1
fi
command -v conda >/dev/null 2>&1 || {
  echo "Conda is required. Install Miniconda or Mambaforge first." >&2
  exit 1
}

echo "This installs the tested Python 3.9 / CUDA 12.8 environment."
echo "Set TRACE_ENV_NAME to override the default environment name."
bash "${ROOT}/envs/build.sh"

echo
echo "Installation complete. Activate with:"
echo "  conda activate ${TRACE_ENV_NAME:-trace}"
