# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: INP001

"""LIT Configuration file for mqt-core-plugins-catalyst.

This file configures the LLVM LIT testing infrastructure for MLIR tests.

Note: `config` and `lit_config` are injected by LIT at runtime.
"""

from __future__ import annotations

from pathlib import Path

import lit.formats
from lit.llvm import llvm_config

# Use `lit_config` to access `config` from lit.site.cfg.py
config = globals().get("config")
if config is None:
    msg = "LIT config object is missing. Ensure lit.site.cfg.py is loaded first."
    raise RuntimeError(msg)

config.name = "MQT Catalyst Plugin Lit Tests"
config.test_format = lit.formats.ShTest(execute_external=False)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir"]

# Define the root path of where to look for tests.
config.test_source_root = Path(__file__).parent

# Define where to execute tests (and produce the output).
config.test_exec_root = Path(config.mqt_core_plugins_catalyst_test_dir)

# Add LLVM tools (FileCheck, not, etc.)
tool_dirs = [config.llvm_tools_dir]
tools = ["not", "FileCheck"]
llvm_config.add_tool_substitutions(tools, tool_dirs)

# Add substitution for the MQT plugin path
config.substitutions.append(("%mqt_plugin_path%", config.mqt_plugin_path))
