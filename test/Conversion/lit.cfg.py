# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""lit configuration for the CatalystQuantum/QCO conversion tests."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import lit.formats

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path

config = globals()["config"]
test_directory = Path(__file__).resolve().parent
project_directory = test_directory.parents[1]
mlir_root_override = os.environ.get("MQT_MLIR_ROOT")
if mlir_root_override is not None:
    mlir_root = Path(mlir_root_override).resolve()
    llvm_revision = None
else:
    llvm_revision = json.loads((project_directory / "toolchain.json").read_text(encoding="utf-8"))["llvm_revision"]
    mlir_root = project_directory / ".cache" / "mlir" / llvm_revision
mlir_bin_directory = mlir_root / "bin"
llvm_revision_file = mlir_root / ".llvm-revision"
plugin_path_override = os.environ.get("MQT_CATALYST_PLUGIN_PATH")
plugin_path = Path(plugin_path_override).resolve() if plugin_path_override else get_catalyst_plugin_abs_path().resolve()

if not mlir_bin_directory.is_dir():
    msg = f"exact cached MLIR toolchain is missing: {mlir_bin_directory}"
    raise RuntimeError(msg)
if mlir_root_override is None:
    found_llvm_revision = (
        llvm_revision_file.read_text(encoding="utf-8").strip() if llvm_revision_file.is_file() else None
    )
    if found_llvm_revision != llvm_revision:
        msg = f"LLVM revision mismatch: expected {llvm_revision}, found {found_llvm_revision}"
        raise RuntimeError(msg)
if not plugin_path.is_file():
    msg = f"Catalyst plugin is missing: {plugin_path}"
    raise RuntimeError(msg)

config.name = "MQT Core Catalyst conversion tests"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py"]
config.test_source_root = str(test_directory)
lit_exec_root = project_directory / ".cache" / "lit"
lit_exec_root.mkdir(parents=True, exist_ok=True)
config.test_exec_root = str(lit_exec_root)

path_entries = [
    str(Path(sys.executable).parent),
    str(mlir_bin_directory),
    config.environment.get("PATH", ""),
]
config.environment["PATH"] = os.pathsep.join(entry for entry in path_entries if entry)
config.substitutions.append(("%mqt_plugin_path%", shlex.quote(str(plugin_path))))
