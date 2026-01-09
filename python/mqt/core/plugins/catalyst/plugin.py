# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for the MQT Catalyst Plugin."""

import site
from importlib.resources import files
from pathlib import Path

__all__ = ["get_catalyst_plugin_abs_path", "name2pass"]


def __dir__() -> list[str]:
    return __all__


def get_catalyst_plugin_abs_path() -> Path:
    """Locate the mqt-catalyst-plugin shared library.

    Returns:
        The absolute path to the plugin shared library.

    Raises:
        FileNotFoundError: If the plugin library is not found.
    """
    # Core library name without platform-specific extensions
    plugin_lib = "mqt-core-plugins-catalyst"

    # Iterate over files in the package directory
    package_path = files("mqt.core.plugins.catalyst")
    for file in package_path.iterdir():
        if file.is_file() and plugin_lib in file.name:
            return Path(str(file))

    # For editable installs, search site-packages directly
    for site_pkg in site.getsitepackages():
        site_pkg_dir = Path(site_pkg) / "mqt" / "core" / "plugins" / "catalyst"
        if site_pkg_dir.exists():
            for file in site_pkg_dir.iterdir():
                if file.is_file() and plugin_lib in file.name:
                    return file

    # Provide helpful error message
    msg = (
        f"Could not locate catalyst plugin library.\n"
        f"Searched for files containing: {plugin_lib}\n"
        f"In package directory: {package_path}\n"
        f"And in site-packages: {site.getsitepackages()}\n"
        f"Ensure the package is properly installed with: pip install -e ."
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
