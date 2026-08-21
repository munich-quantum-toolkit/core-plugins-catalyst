#!/usr/bin/env bash
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly SCRIPT_DIR
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
readonly PROJECT_DIR
readonly TOOLCHAIN_PINS="${PROJECT_DIR}/toolchain.json"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.cache/uv}"

for required_tool in uv git curl make; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    echo "${required_tool} is required to bootstrap the project." >&2
    exit 1
  fi
done

IFS=$'\t' read -r \
  CATALYST_VERSION \
  CATALYST_REVISION \
  LLVM_REVISION \
  SETUP_MLIR_VERSION \
  SETUP_MLIR_SHA256 \
  BOOTSTRAP_REVISION < <(
    uv run --no-project python - "${TOOLCHAIN_PINS}" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys

pins = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    pins["catalyst_version"],
    pins["catalyst_revision"],
    pins["llvm_revision"],
    pins["setup_mlir_version"],
    pins["setup_mlir_sha256"],
    pins["bootstrap_revision"],
    sep="\t",
)
PY
  )
readonly CATALYST_VERSION CATALYST_REVISION LLVM_REVISION SETUP_MLIR_VERSION
readonly SETUP_MLIR_SHA256 BOOTSTRAP_REVISION
readonly MLIR_ROOT="${PROJECT_DIR}/.cache/mlir/${LLVM_REVISION}"
readonly CATALYST_ROOT="${PROJECT_DIR}/.cache/catalyst/${CATALYST_REVISION}"
readonly CATALYST_BUILD_VENV="${PROJECT_DIR}/.cache/catalyst-build/${CATALYST_REVISION}/cp312"
readonly BUILD_PYTHON_EXECUTABLE="${CATALYST_BUILD_VENV}/bin/python"
readonly LLVM_REVISION_FILE="${MLIR_ROOT}/.llvm-revision"
readonly CATALYST_BUILD_MARKER="${CATALYST_ROOT}/.mqt-bootstrap-complete"
readonly MLIR_PYTHON_PATCH="${PROJECT_DIR}/cmake/patches/catalyst-mlir-python-sources-only.patch"
readonly CATALYST_REVERSE_ENUMERATE_PATCH="${PROJECT_DIR}/cmake/patches/catalyst-libstdcxx14-reverse-enumerate.patch"

if [[ -n "${MQT_BOOTSTRAP_PYTHON:-}" ]]; then
  PYTHON_EXECUTABLE="${MQT_BOOTSTRAP_PYTHON}"
else
  readonly VENV_DIR="${PROJECT_DIR}/.venv"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    uv venv --python 3.12 "${VENV_DIR}"
  fi
  PYTHON_EXECUTABLE="${VENV_DIR}/bin/python"
fi
readonly PYTHON_EXECUTABLE

python_version="$(uv run --no-project --python "${PYTHON_EXECUTABLE}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${python_version}" in
  3.12 | 3.13 | 3.14) ;;
  *)
    echo "Catalyst ${CATALYST_VERSION} requires Python 3.12, 3.13, or 3.14; found ${python_version}." >&2
    exit 1
    ;;
esac

mkdir -p "${MLIR_ROOT}"
if [[ -f "${MLIR_ROOT}/lib/cmake/mlir/MLIRConfig.cmake" && -f "${LLVM_REVISION_FILE}" ]]; then
  found_llvm_revision="$(<"${LLVM_REVISION_FILE}")"
  if [[ "${found_llvm_revision}" != "${LLVM_REVISION}" ]]; then
    echo "Cached LLVM revision mismatch: expected ${LLVM_REVISION}, found ${found_llvm_revision}." >&2
    exit 1
  fi
else
  setup_mlir_script="$(mktemp "${TMPDIR:-/tmp}/setup-mlir.XXXXXX")"
  trap 'rm -f -- "${setup_mlir_script}"' EXIT
  curl -LsSf \
    "https://github.com/munich-quantum-software/setup-mlir/releases/download/v${SETUP_MLIR_VERSION}/setup-mlir.sh" \
    -o "${setup_mlir_script}"
  found_setup_mlir_sha256="$(
    uv run --no-project python -c \
      'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
      "${setup_mlir_script}"
  )"
  if [[ "${found_setup_mlir_sha256}" != "${SETUP_MLIR_SHA256}" ]]; then
    echo "setup-mlir checksum mismatch: expected ${SETUP_MLIR_SHA256}, found ${found_setup_mlir_sha256}." >&2
    exit 1
  fi
  bash "${setup_mlir_script}" -v "${LLVM_REVISION}" -p "${MLIR_ROOT}"
  rm -f -- "${setup_mlir_script}"
  trap - EXIT
  printf '%s\n' "${LLVM_REVISION}" >"${LLVM_REVISION_FILE}"
fi

if [[ ! -f "${MLIR_ROOT}/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
  echo "setup-mlir did not create a usable MLIR installation at ${MLIR_ROOT}." >&2
  exit 1
fi

if [[ ! -d "${CATALYST_ROOT}/.git" ]]; then
  if [[ -e "${CATALYST_ROOT}" ]]; then
    echo "Removing incomplete Catalyst cache: ${CATALYST_ROOT}" >&2
    rm -rf -- "${CATALYST_ROOT}"
  fi
  mkdir -p "$(dirname -- "${CATALYST_ROOT}")"
  catalyst_clone_dir="$(mktemp -d "$(dirname -- "${CATALYST_ROOT}")/.catalyst-clone.XXXXXX")"
  if ! git clone --filter=blob:none --no-checkout https://github.com/PennyLaneAI/catalyst.git \
    "${catalyst_clone_dir}" \
    || ! git -C "${catalyst_clone_dir}" checkout --detach "${CATALYST_REVISION}"; then
    rm -rf -- "${catalyst_clone_dir}"
    echo "Unable to clone Catalyst ${CATALYST_REVISION}; the temporary checkout was removed." >&2
    exit 1
  fi
  mv "${catalyst_clone_dir}" "${CATALYST_ROOT}"
fi

found_catalyst_revision="$(git -C "${CATALYST_ROOT}" rev-parse HEAD)"
if [[ "${found_catalyst_revision}" != "${CATALYST_REVISION}" ]]; then
  echo "Cached Catalyst revision mismatch: expected ${CATALYST_REVISION}, found ${found_catalyst_revision}. Remove ${CATALYST_ROOT} and rerun the bootstrap." >&2
  exit 1
fi

if git -C "${CATALYST_ROOT}" apply --reverse --check \
  "${CATALYST_REVERSE_ENUMERATE_PATCH}" >/dev/null 2>&1; then
  : # The bootstrap patch is already applied.
elif git -C "${CATALYST_ROOT}" apply --check \
  "${CATALYST_REVERSE_ENUMERATE_PATCH}" >/dev/null 2>&1; then
  git -C "${CATALYST_ROOT}" apply "${CATALYST_REVERSE_ENUMERATE_PATCH}"
else
  echo "Unable to apply the Catalyst reverse-enumeration bootstrap patch." >&2
  exit 1
fi

git -C "${CATALYST_ROOT}" submodule update --init --recursive --depth 1
found_catalyst_llvm_revision="$(git -C "${CATALYST_ROOT}/mlir/llvm-project" rev-parse HEAD)"
if [[ "${found_catalyst_llvm_revision}" != "${LLVM_REVISION}" ]]; then
  echo "Catalyst pins LLVM ${found_catalyst_llvm_revision}, expected ${LLVM_REVISION}." >&2
  exit 1
fi

if git -C "${CATALYST_ROOT}/mlir/llvm-project" apply --reverse --check \
  "${MLIR_PYTHON_PATCH}" >/dev/null 2>&1; then
  : # The bootstrap patch is already applied.
elif git -C "${CATALYST_ROOT}/mlir/llvm-project" apply --check \
  "${MLIR_PYTHON_PATCH}" >/dev/null 2>&1; then
  git -C "${CATALYST_ROOT}/mlir/llvm-project" apply "${MLIR_PYTHON_PATCH}"
else
  echo "Unable to apply the Catalyst MLIR Python-source bootstrap patch." >&2
  exit 1
fi

found_declared_llvm_revision="$(awk -F= '$1 == "llvm" { print $2 }' "${CATALYST_ROOT}/.dep-versions")"
if [[ "${found_declared_llvm_revision}" != "${LLVM_REVISION}" ]]; then
  echo "Catalyst .dep-versions pins LLVM ${found_declared_llvm_revision}, expected ${LLVM_REVISION}." >&2
  exit 1
fi

catalyst_wheel=""
if [[ -d "${CATALYST_ROOT}/dist" ]]; then
  catalyst_wheel_count="$(find "${CATALYST_ROOT}/dist" -maxdepth 1 -type f -name '*.whl' -print | wc -l | tr -d ' ')"
  if [[ "${catalyst_wheel_count}" == 1 ]]; then
    catalyst_wheel="$(find "${CATALYST_ROOT}/dist" -maxdepth 1 -type f -name '*.whl' -print -quit)"
  fi
fi
expected_build_marker="${CATALYST_REVISION}:${LLVM_REVISION}:cp312-abi3:setup-mlir-v${SETUP_MLIR_VERSION}:bootstrap-v${BOOTSTRAP_REVISION}"
found_build_marker=""
if [[ -f "${CATALYST_BUILD_MARKER}" ]]; then
  found_build_marker="$(<"${CATALYST_BUILD_MARKER}")"
fi
if [[ "${found_build_marker}" != "${expected_build_marker}" || -z "${catalyst_wheel}" ]]; then
  if [[ ! -x "${BUILD_PYTHON_EXECUTABLE}" ]]; then
    uv venv --python 3.12 "${CATALYST_BUILD_VENV}"
  fi
  build_python_version="$(
    uv run --no-project --python "${BUILD_PYTHON_EXECUTABLE}" \
      python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
  )"
  if [[ "${build_python_version}" != "3.12" ]]; then
    echo "Catalyst's native wheel must be built with Python 3.12; found ${build_python_version}." >&2
    exit 1
  fi

  uv pip install --python "${BUILD_PYTHON_EXECUTABLE}" \
    "cmake>=3.26,<4" \
    lit \
    ninja \
    "numpy>2.0.0" \
    "nanobind>=2.9,<2.13" \
    "pybind11>=2.12.0" \
    PyYAML \
    "scikit-build-core~=0.12" \
    setuptools \
    "setuptools-scm>=9.2.2" \
    "pip>=22.3" \
    wheel

  # Install the Python dependencies and a temporary frontend build. The
  # complete source wheel below replaces it after the native components have
  # been built.
  installed_build_revision="$(
    "${BUILD_PYTHON_EXECUTABLE}" - <<'PY'
from __future__ import annotations

import importlib.metadata
import runpy

try:
    distribution = importlib.metadata.distribution("pennylane-catalyst")
    revision = runpy.run_path(distribution.locate_file("catalyst/_revision.py"))["__revision__"]
except (FileNotFoundError, importlib.metadata.PackageNotFoundError, KeyError):
    pass
else:
    print(revision)
PY
  )"
  if [[ "${installed_build_revision}" != "${CATALYST_REVISION}" ]]; then
    uv pip install --python "${BUILD_PYTHON_EXECUTABLE}" \
      --extra-index-url https://test.pypi.org/simple \
      --index-strategy unsafe-best-match \
      --prerelease allow \
      --reinstall "${CATALYST_ROOT}"
  fi

  export PYTHON="${BUILD_PYTHON_EXECUTABLE}"
  export PATH="$(dirname -- "${BUILD_PYTHON_EXECUTABLE}"):${MLIR_ROOT}/bin:${PATH}"
  for build_tool in cmake ninja; do
    if ! command -v "${build_tool}" >/dev/null 2>&1; then
      echo "${build_tool} is required to build Catalyst." >&2
      exit 1
    fi
  done
  export CMAKE_PREFIX_PATH="${MLIR_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export CCACHE_DIR="${PROJECT_DIR}/.cache/ccache"
  export CCACHE_TEMPDIR="${PROJECT_DIR}/.cache/ccache/tmp"
  export SCCACHE_DIR="${PROJECT_DIR}/.cache/sccache"
  mkdir -p "${CCACHE_TEMPDIR}"
  export LLVM_BUILD_DIR="${MLIR_ROOT}"
  export LLVM_INSTALL_STAGING_DIR="${MLIR_ROOT}"
  export LLVM_EXTERNAL_LIT="$(dirname -- "${BUILD_PYTHON_EXECUTABLE}")/lit"
  export LLVM_DIR="${CATALYST_ROOT}/mlir/llvm-project"
  export STABLEHLO_BUILD_DIR="${CATALYST_ROOT}/mlir/stablehlo/build"
  export ENZYME_BUILD_DIR="${CATALYST_ROOT}/mlir/Enzyme/build"
  export DIALECTS_BUILD_DIR="${CATALYST_ROOT}/mlir/build"
  export RT_BUILD_DIR="${CATALYST_ROOT}/runtime/build"
  export OQC_BUILD_DIR="${CATALYST_ROOT}/frontend/catalyst/third_party/oqc/src/build"

  compiler_launcher="$(command -v sccache || command -v ccache || true)"
  c_compiler="${CC:-}"
  cxx_compiler="${CXX:-}"
  if [[ -z "${c_compiler}" ]]; then
    c_compiler="$(command -v clang || command -v cc || true)"
  fi
  if [[ -z "${cxx_compiler}" ]]; then
    cxx_compiler="$(command -v clang++ || command -v c++ || true)"
  fi
  if [[ -z "${c_compiler}" || -z "${cxx_compiler}" ]]; then
    echo "A C compiler and a C++ compiler are required to build Catalyst." >&2
    exit 1
  fi
  export C_COMPILER="${c_compiler}"
  export CXX_COMPILER="${cxx_compiler}"
  export COMPILER_LAUNCHER="${compiler_launcher}"
  if [[ -z "${FC:-}" ]]; then
    for fortran_name in gfortran gfortran-14 gfortran-13 gfortran-15; do
      if fortran_compiler="$(command -v "${fortran_name}" 2>/dev/null)"; then
        export FC="${fortran_compiler}"
        break
      fi
    done
  fi
  if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 && -z "${FC:-}" ]]; then
    echo "A Fortran compiler is required to build Catalyst's macOS runtime." >&2
    exit 1
  fi

  enable_lld=OFF
  if command -v ld.lld >/dev/null 2>&1; then
    enable_lld=ON
  fi

  # Deliberately do not invoke Catalyst's llvm target. Every native component
  # is built against the setup-mlir installation above.
  make -C "${CATALYST_ROOT}" stablehlo ENABLE_LLD="${enable_lld}"
  make -C "${CATALYST_ROOT}" enzyme
  make -C "${CATALYST_ROOT}" runtime

  cmake -G Ninja -S "${CATALYST_ROOT}/mlir" -B "${DIALECTS_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${c_compiler}" \
    -DCMAKE_CXX_COMPILER="${cxx_compiler}" \
    -DCMAKE_C_COMPILER_LAUNCHER="${compiler_launcher}" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${compiler_launcher}" \
    -DCMAKE_PROJECT_Catalyst_INCLUDE="${PROJECT_DIR}/cmake/CatalystMLIRPythonSources.cmake" \
    -DEnzyme_DIR="${ENZYME_BUILD_DIR}" \
    -DENZYME_SRC_DIR="${CATALYST_ROOT}/mlir/Enzyme" \
    -DLLVM_DIR="${MLIR_ROOT}/lib/cmake/llvm" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_LLD="${enable_lld}" \
    -DLLVM_ENABLE_ZLIB=ON \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT}" \
    -DLLVM_USE_SANITIZER= \
    -DMLIR_DIR="${MLIR_ROOT}/lib/cmake/mlir" \
    -DMLIR_LIB_DIR="${MLIR_ROOT}/lib" \
    -DMQT_CATALYST_LLVM_SOURCE_DIR="${CATALYST_ROOT}/mlir/llvm-project" \
    -DPython_EXECUTABLE="${BUILD_PYTHON_EXECUTABLE}" \
    -DPython3_EXECUTABLE="${BUILD_PYTHON_EXECUTABLE}" \
    -DPython3_NumPy_INCLUDE_DIRS="$(
      "${BUILD_PYTHON_EXECUTABLE}" -c 'import numpy; print(numpy.get_include())'
    )" \
    -DQUANTUM_ENABLE_BINDINGS_PYTHON=ON \
    -DRUNTIME_LIB_DIR="${RT_BUILD_DIR}/lib" \
    -DSTABLEHLO_BUILD_DIR="${STABLEHLO_BUILD_DIR}" \
    -DSTABLEHLO_DIR="${CATALYST_ROOT}/mlir/stablehlo" \
    -DCATALYST_ENABLE_WARNINGS=ON
  cmake --build "${DIALECTS_BUILD_DIR}" --target \
    catalyst-cli \
    QuantumPythonModules

  # Catalyst normally patches mlir-tblgen before building LLVM. setup-mlir's
  # prebuilt tblgen is intentionally reused here, so apply the equivalent
  # future-annotations fix to its generated Python operation bindings.
  "${BUILD_PYTHON_EXECUTABLE}" - "${DIALECTS_BUILD_DIR}" <<'PY'
from __future__ import annotations

import pathlib
import sys

root = pathlib.Path(sys.argv[1])
marker = "from __future__ import annotations"
header = "# Autogenerated by mlir-tblgen; don't manually edit.\n"
generated = list(root.rglob("*_ops_gen.py"))
if not generated:
    raise SystemExit(f"No generated MLIR Python operation bindings found below {root}")
for path in generated:
    source = path.read_text()
    if marker not in source:
        if header not in source:
            raise SystemExit(f"Unexpected generated binding header: {path}")
        path.write_text(source.replace(header, f"{header}\n{marker}\n", 1))
PY

  make -C "${CATALYST_ROOT}" oqc
  # The bootstrap wheel must replace any same-version temporary frontend.
  # Otherwise pip retains that incomplete install and precompilation cannot
  # import the generated mlir_quantum package.
  uv pip uninstall --python "${BUILD_PYTHON_EXECUTABLE}" pennylane-catalyst
  if [[ -d "${CATALYST_ROOT}/dist" ]]; then
    rm -rf -- "${CATALYST_ROOT}/dist"
  fi
  make -C "${CATALYST_ROOT}" wheel

  catalyst_wheel_count=0
  if [[ -d "${CATALYST_ROOT}/dist" ]]; then
    catalyst_wheel_count="$(find "${CATALYST_ROOT}/dist" -maxdepth 1 -type f -name '*.whl' -print | wc -l | tr -d ' ')"
  fi
  if [[ "${catalyst_wheel_count}" != 1 ]]; then
    echo "Catalyst must produce exactly one wheel; found ${catalyst_wheel_count}." >&2
    exit 1
  fi
  catalyst_wheel="$(find "${CATALYST_ROOT}/dist" -maxdepth 1 -type f -name '*.whl' -print -quit)"
  printf '%s\n' "${expected_build_marker}" >"${CATALYST_BUILD_MARKER}"
fi
uv pip install --python "${PYTHON_EXECUTABLE}" \
  --extra-index-url https://test.pypi.org/simple \
  --index-strategy unsafe-best-match \
  --prerelease allow \
  --reinstall-package pennylane-catalyst \
  "${catalyst_wheel}"

uv run --no-project --python "${PYTHON_EXECUTABLE}" python - \
  "${CATALYST_VERSION}" "${CATALYST_REVISION}" "${LLVM_REVISION}" <<'PY'
from __future__ import annotations

import pathlib
import sys

import catalyst
from catalyst.utils.runtime_environment import get_include_path

expected_version, expected_revision, llvm_revision = sys.argv[1:]
if catalyst.__version__ != expected_version:
    raise SystemExit(f"Catalyst version mismatch: {catalyst.__version__} != {expected_version}")
if catalyst.__revision__ != expected_revision:
    raise SystemExit(f"Catalyst revision mismatch: {catalyst.__revision__} != {expected_revision}")
if not pathlib.Path(get_include_path()).is_dir():
    raise SystemExit(f"Catalyst include directory does not exist: {get_include_path()}")
print(
    f"Verified Catalyst {expected_version} ({expected_revision}), "
    f"and LLVM {llvm_revision}."
)
PY

echo "MLIR_DIR=${MLIR_ROOT}/lib/cmake/mlir"
echo "Python_EXECUTABLE=${PYTHON_EXECUTABLE}"
