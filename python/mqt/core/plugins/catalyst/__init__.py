# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Catalyst Plugin."""

from __future__ import annotations

import platform
from importlib import resources
from pathlib import Path

from mqt.core.plugins.catalyst.device import configure_device_for_mqt, get_device


def get_catalyst_plugin_abs_path() -> Path:
    """Locate the mqt-catalyst-plugin shared library.

    Returns:
        The absolute path to the plugin shared library.

    Raises:
        FileNotFoundError: If the plugin library is not found.
        RuntimeError: If the platform is unsupported.
    """
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(platform.system())
    if ext is None:
        msg = f"Unsupported platform: {platform.system()}"
        raise RuntimeError(msg)

    # Try to find the plugin library in the package installation directory
    try:
        if hasattr(resources, "files"):
            package_path = resources.files("mqt.core.plugins.catalyst")
            # Try both with and without lib prefix
            for lib_name in [f"mqt-core-plugins-catalyst{ext}", f"libmqt-core-plugins-catalyst{ext}"]:
                lib_path = package_path / lib_name
                if lib_path.is_file():
                    return Path(str(lib_path))
    except (AttributeError, TypeError, FileNotFoundError, ModuleNotFoundError):
        # Fall back to development build directory if package resources unavailable
        pass

    # Fallback: search in development build directory (for editable installs)
    this_file = Path(__file__).resolve()
    # python/mqt/core/plugins/catalyst/__init__.py -> go up 6 levels to project root
    project_root = this_file.parent.parent.parent.parent.parent.parent
    build_dir = project_root / "build"

    if build_dir.exists():
        # Try both with and without lib prefix
        for lib_name in [f"mqt-core-plugins-catalyst{ext}", f"libmqt-core-plugins-catalyst{ext}"]:
            # Search recursively in build directory
            for lib_path in build_dir.rglob(lib_name):
                return lib_path

    # Provide helpful error message
    lib_names = [f"mqt-core-plugins-catalyst{ext}", f"libmqt-core-plugins-catalyst{ext}"]
    msg = (
        f"Could not locate catalyst plugin library with extension '{ext}'.\n"
        f"Searched for: {', '.join(lib_names)}\n"
        f"Could not locate catalyst plugin library with extension '{ext}'.\n"
        f"Expected locations:\n"
        f"  - Installed package: {this_file.parent}\n"
        f"  - Development build: {build_dir}\n"
        f"For editable install, ensure the library is built: cmake --build build"
    )
    raise FileNotFoundError(msg)


def name2pass(name: str) -> tuple[Path, str]:
    """Convert a pass name to its plugin path and pass name (required by Catalyst).

    Args:
        name: The name of the pass, e.g., "mqt-core-round-trip".

    Returns:
        A tuple containing the absolute path to the plugin and the pass name.
    """
    return get_catalyst_plugin_abs_path(), name


__all__ = [
    "configure_device_for_mqt",
    "get_catalyst_plugin_abs_path",
    "get_device",
    "name2pass",
]
