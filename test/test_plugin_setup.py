# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MQT plugin setup with PennyLane and Catalyst.

These tests only check that the MQT plugin is correctly installed and
can be used in various ways with PennyLane (they do NOT execute any pass).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pennylane as qml
import pytest
from catalyst import pipeline
from catalyst.passes import apply_pass, apply_pass_plugin

from mqt.core.plugins.catalyst import configure_device_for_mqt, get_catalyst_plugin_abs_path, get_device

if TYPE_CHECKING:
    from pennylane.measurements.state import StateMP


def test_mqt_plugin() -> None:
    """Generate MLIR for the MQT plugin.

    Does not execute the pass.
    """
    plugin_path = str(get_catalyst_plugin_abs_path())

    @apply_pass("mqt-core-round-trip")  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("null.qubit", wires=0))  # type: ignore[untyped-decorator]
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(pass_plugins={plugin_path}, dialect_plugins={plugin_path}, target="mlir")  # type: ignore[untyped-decorator]
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_plugin_no_preregistration() -> None:
    """Generate MLIR for the MQT plugin.

    No need to register the plugin ahead of time in the qjit decorator.
    """
    plugin_path = str(get_catalyst_plugin_abs_path())

    @apply_pass_plugin(plugin_path, "mqt-core-round-trip")  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("null.qubit", wires=0))  # type: ignore[untyped-decorator]
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")  # type: ignore[untyped-decorator]
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_entry_point() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    @apply_pass("mqt.mqt-core-round-trip")  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("null.qubit", wires=0))  # type: ignore[untyped-decorator]
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")  # type: ignore[untyped-decorator]
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_dictionary() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    @pipeline({"mqt.mqt-core-round-trip": {}})  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("null.qubit", wires=0))  # type: ignore[untyped-decorator]
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")  # type: ignore[untyped-decorator]
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_get_catalyst_plugin_abs_path_unsupported_platform() -> None:
    """Test that get_catalyst_plugin_abs_path raises RuntimeError on unsupported platform."""
    with (
        patch("platform.system", return_value="UnsupportedOS"),
        pytest.raises(RuntimeError, match="Unsupported platform: UnsupportedOS"),
    ):
        get_catalyst_plugin_abs_path()


def test_get_catalyst_plugin_abs_path_not_found() -> None:
    """Test that get_catalyst_plugin_abs_path raises FileNotFoundError when library is missing."""
    with (
        patch("mqt.core.plugins.catalyst.resources.files", side_effect=Exception("Not found")),
        patch("pathlib.Path.exists", return_value=False),
        pytest.raises(FileNotFoundError, match="Could not locate catalyst plugin library"),
    ):
        get_catalyst_plugin_abs_path()


def test_configure_device_for_mqt_no_config() -> None:
    """Test that configure_device_for_mqt raises ValueError when device has no config_filepath."""
    dev = MagicMock(spec=qml.devices.Device)
    dev.config_filepath = None
    with pytest.raises(ValueError, match=r"Device does not have a config_filepath attribute set\."):
        configure_device_for_mqt(dev)


def test_get_device_no_config() -> None:
    """Test that get_device raises ValueError when the created device has no config_filepath."""
    with patch("pennylane.device") as mock_qml_device:
        mock_dev = MagicMock(spec=qml.devices.Device)
        mock_dev.config_filepath = None
        mock_qml_device.return_value = mock_dev

        with pytest.raises(ValueError, match=r"Device does not have a config_filepath attribute set\."):
            get_device("some.device")
