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
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.cache/uv}"
llvm_revision="$(
  uv run --no-project python -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["llvm_revision"])' \
    "${PROJECT_DIR}/toolchain.json"
)"
export MQT_MLIR_ROOT="${PROJECT_DIR}/.cache/mlir/${llvm_revision}"
exec lit "$@"
