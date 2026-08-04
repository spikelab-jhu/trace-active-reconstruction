#!/usr/bin/env bash
# NOTE: no `-u`. conda's gcc_linux-64 deactivate hook dereferences
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED, which aborts the install
# under `set -u` on a clean environment.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${TRACE_ENV_NAME:-trace}"
CUDA_VERSION="${TRACE_CUDA_VERSION:-12.8}"

CONDA_PATH="$(conda info --base)"
source "${CONDA_PATH}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.9 cmake=3.14.0
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=True

conda install -y -c nvidia "cuda-toolkit=${CUDA_VERSION}"
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# RTX 50-series support requires CUDA 12.8 compatible PyTorch wheels.
python -m pip install --upgrade pip wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja
python -m pip install -r "${ROOT}/envs/requirements.txt"

# Habitat-Sim fork used by the original ActiveGS implementation.
HABITAT_DIR="${ROOT}/simulator/habitat-sim"
if [[ ! -d "${HABITAT_DIR}/.git" ]]; then
  git clone https://github.com/liren-jin/habitat-sim.git "${HABITAT_DIR}"
fi
python -m pip install -r "${HABITAT_DIR}/requirements.txt"
(
  cd "${HABITAT_DIR}"
  # Build habitat-sim with the SYSTEM compilers. `conda install cuda-toolkit`
  # pulls in gcc_linux-64/sysroot_linux-64 and its activation scripts export
  # CC/CXX pointing at the conda toolchain; cmake then searches only the conda
  # sysroot and fails with "Could NOT find OpenGL (missing: OPENGL_opengl_LIBRARY
  # OPENGL_glx_LIBRARY)" even though the system libGL/libGLX are installed.
  # cstdint-prelude.h: GCC 13+ dropped transitive <cstdint> includes; the
  # vendored corrade/magnum predate that and fail on std::uint32_t otherwise.
  # The guarded header (not a bare `-include cstdint`) keeps CMake's c++98
  # feature test compiling, see the header comment.
  env -u CC -u CXX -u CMAKE_ARGS -u CMAKE_PREFIX_PATH \
      -u LDFLAGS -u CFLAGS -u CXXFLAGS -u CPPFLAGS \
      CC=/usr/bin/cc CXX=/usr/bin/c++ \
      CXXFLAGS="-include ${ROOT}/envs/cstdint-prelude.h" \
      python setup.py install --headless --bullet
)

# Differentiable Gaussian rasterizer. CUDA 12.8 needs explicit cstdint includes.
DGR_DIR="${ROOT}/simulator/diff-gaussian-rasterization_2d"
if [[ ! -d "${DGR_DIR}/.git" ]]; then
  git clone --recursive \
    https://github.com/liren-jin/diff-gaussian-rasterization_2d \
    "${DGR_DIR}"
fi
for file in \
  "${DGR_DIR}/cuda_rasterizer/backward.h" \
  "${DGR_DIR}/cuda_rasterizer/rasterizer_impl.cu" \
  "${DGR_DIR}/cuda_rasterizer/rasterizer_impl.h" \
  "${DGR_DIR}/cuda_rasterizer/forward.h" \
  "${DGR_DIR}/cuda_rasterizer/forward.cu" \
  "${DGR_DIR}/cuda_rasterizer/backward.cu"; do
  grep -q "include <cstdint>" "${file}" || sed -i '1i #include <cstdint>' "${file}"
done
# Same system-compiler rule as habitat-sim: conda gcc 14 emits calls needing
# CXXABI_1.3.15, but the libstdc++ actually loaded at runtime (system, and the
# one conda ships) only provides 1.3.9, so the extension imports with
# "version `CXXABI_1.3.15' not found". Building against the system toolchain
# keeps the whole environment ABI-consistent.
env -u CC -u CXX -u CMAKE_ARGS -u CMAKE_PREFIX_PATH \
    -u LDFLAGS -u CFLAGS -u CXXFLAGS -u CPPFLAGS \
    CC=/usr/bin/cc CXX=/usr/bin/c++ \
    python -m pip install --no-build-isolation "${DGR_DIR}"

echo "Environment ${ENV_NAME} is ready."
