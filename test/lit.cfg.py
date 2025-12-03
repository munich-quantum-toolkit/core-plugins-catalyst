# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: INP001

"""LIT Configuration file for the MQT Core Catalyst Plugin.

This file configures the LLVM LIT testing infrastructure for MLIR tests.
Run tests with: lit test

Note: `config` and `lit_config` are injected by LIT at runtime.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import lit.formats
import lit.llvm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent


def find_build_dir() -> Path:
    """Find the most recent build directory."""
    build_root = PROJECT_ROOT / "build"
    if not build_root.exists():
        msg = f"Build directory not found at {build_root}. Run 'uv pip install -e . --no-build-isolation' first."
        raise FileNotFoundError(msg)

    # Find build directories with Release/Debug configs
    build_dirs = []
    for subdir in build_root.iterdir():
        if subdir.is_dir():
            # Check for nested structure (e.g., build/cp312-.../Release)
            build_dirs.extend(
                nested for nested in subdir.iterdir() if nested.is_dir() and nested.name in {"Release", "Debug"}
            )

    if not build_dirs:
        msg = f"No build configuration found in {build_root}. Run 'uv pip install -e . --no-build-isolation' first."
        raise FileNotFoundError(msg)

    # Return the most recently modified build directory
    return max(build_dirs, key=lambda d: d.stat().st_mtime)


def find_llvm_tools_dir() -> str:
    """Find LLVM tools directory."""
    # Check environment variable first
    if llvm_dir := os.environ.get("LLVM_TOOLS_DIR", os.environ.get("LLVM_DIR", "")):
        return llvm_dir

    # Try to find from catalyst installation
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import catalyst; print(catalyst.__path__[0])"],
            capture_output=True,
            text=True,
            check=True,
        )
        catalyst_path = Path(result.stdout.strip())
        llvm_bin = catalyst_path / "bin"
        if llvm_bin.exists() and (llvm_bin / "FileCheck").exists():
            return str(llvm_bin)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check common locations
    common_paths = [
        Path.home() / "Code" / "llvm-project" / "build" / "bin",
        Path("/opt/homebrew/opt/llvm/bin"),
        Path("/usr/local/opt/llvm/bin"),
    ]
    for check_path in common_paths:
        if check_path.exists() and (check_path / "FileCheck").exists():
            return str(check_path)

    return ""


# Find build directory and configure paths
build_dir = find_build_dir()
llvm_tools = find_llvm_tools_dir()
plugin_ext = ".dylib" if platform.system() == "Darwin" else ".so"
plugin_path = build_dir / "lib" / f"mqt-core-catalyst-plugin{plugin_ext}"

# LIT Configuration
config.name = "MQT Core Catalyst Plugin"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = str(TEST_DIR)
config.test_exec_root = str(build_dir / "test")

# Ensure test output directory exists
(build_dir / "test").mkdir(parents=True, exist_ok=True)

# Initialize LLVM LIT support
config.llvm_tools_dir = llvm_tools
lit.llvm.initialize(lit_config, config)

# Import llvm_config after initialization
from lit.llvm import llvm_config

# Add tool substitutions
tool_dirs = [llvm_tools, str(build_dir / "lib")]
llvm_config.add_tool_substitutions(["FileCheck", "not"], tool_dirs)

# Add plugin path substitution
config.substitutions.append(("%mqt_plugin_path%", str(plugin_path)))
